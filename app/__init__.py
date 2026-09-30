"""FastAPI main app"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from agents.responder import responder_agent
from app import utils
from config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
queue = asyncio.Queue()


class UserInput(BaseModel):
    """用户输入"""

    query: str
    user_id: str

    @field_validator("query")
    def validate_query_has_letters(cls, value):
        """验证查询至少包含一个字母或汉字"""
        if not re.search(r"[a-zA-Z\u4e00-\u9fff]", value):
            raise ValueError("查询不能为空，且至少包含一个字母、数字或汉字")
        return value

async def enqueue_query(user_input: UserInput) -> None:
    """将用户输入加入队列"""
    await queue.put(user_input)


async def process_user_input() -> dict:
    """处理用户输入"""
    while True:
        user_input = await queue.get()
        logger.info(" * 正在处理用户输入: %s", user_input)
        result = responder_agent.invoke(
            {"messages": [{"role": "user", "content": user_input.query}]}
        )
        query_response = result.get("messages")[-1].content
        async with httpx.AsyncClient() as client:
            try:
                webhook_response = await client.post(
                    url=settings.CALLBACK_URL,
                    json={"query": user_input.query, "response": query_response},
                )
                logger.info(" * WEBHOOK 响应: %s", webhook_response)
            except Exception as error:
                logger.error(error)
        logger.info(" * RAG 响应: %s", query_response)
        queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期"""
    logger.info(" * %s 应用已启动", app.title)
    logger.info(" * 正在 RAG 管道中索引文档")
    utils.index_documents()
    task = asyncio.create_task(process_user_input())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, title="RAG 多 Agent 聊天机器人")


@app.post("/query")
async def user_query(user_input: UserInput, background_tasks: BackgroundTasks):
    """处理用户查询"""
    # 一旦生成 RAG 响应，发送到 webhook
    try:
        background_tasks.add_task(enqueue_query, user_input)
        logger.info(" * 用户查询已成功加入队列！")
        return {"message": "用户查询已成功加入队列！"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/webhook")
async def receive_callback(data: dict, request: Request):
    logger.info("\n--- 收到回调 ---")
    logger.info("来自: %s:%s", request.client.host, request.client.port)
    logger.info("回调数据: %s\n-------------------------", data)
    return {"status": "success", "message": "回调已接收"}


if __name__ == "__main__":
    ...
