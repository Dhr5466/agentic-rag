"""FastAPI main app"""
# 标准库
import asyncio
import contextlib
import logging
import re
from contextlib import asynccontextmanager

# 第三方
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
#本地模块
from config import settings
from app import utils
from agents.responder import responder_agent


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
queue = asyncio.Queue()
# 全局 HTTP 客户端，在 lifespan 中创建 / 关闭
http_client: httpx.AsyncClient | None = None
"""logging.basicConfig(...)：设置日志等级和输出格式。

logger：后续通过 logger.info()、logger.error() 记录运行状态。

queue = asyncio.Queue()：创建一个只存在于当前 Python 进程内存中的队列。"""

class UserInput(BaseModel):
    """用户输入"""
    query: str
    user_id: str
    @field_validator("query")
    def validate_query_has_letters(cls, value):
        """验证查询至少包含一个字母或汉字"""
        if not re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", value):
            raise ValueError("查询不能为空，且至少包含一个字母、数字或汉字")
        return value

async def enqueue_query(user_input: UserInput) -> None:
    """将用户输入加入队列"""
    await queue.put(user_input)


async def process_user_input() -> dict:
    """从队列中不断取任务并处理；单个任务失败不影响后续任务"""
    while True:
        user_input = await queue.get()
        try:
            logger.info(" * 正在处理用户输入: %s", user_input)

            # responder_agent.invoke 是同步阻塞，丢线程池
            result = await asyncio.to_thread(
                responder_agent.invoke,
                {"messages": [{"role": "user", "content": user_input.query}]},
            )
            query_response = result.get("messages")[-1].content
            logger.info(" * RAG 响应: %s", query_response)

            # 发送 webhook
            try:
                webhook_response = await http_client.post(
                    url=settings.CALLBACK_URL,
                    json={"query": user_input.query, "response": query_response},
                )
                logger.info(" * WEBHOOK 响应状态: %s", webhook_response.status_code)
            except Exception:
                # webhook 失败不影响这一轮的主流程
                logger.exception(" * WEBHOOK 发送失败")

        except asyncio.CancelledError:
            # 收到关闭信号，让 task_done 也执行（可选），然后重新抛出让外部感知
            logger.info(" * process_user_input 收到取消信号，退出")
            raise
        except Exception:
            logger.exception(" * 处理用户输入失败")
        finally:
            # 无论成功失败，都标记这一条完成
            queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 应用生命周期"""
    global http_client

    # ========== 启动 ==========
    logger.info(" * %s 应用已启动", app.title)

    logger.info(" * 正在 RAG 管道中索引文档")
    try:
        await asyncio.to_thread(utils.index_documents)
    except Exception:
        logger.exception("RAG 索引失败")

    http_client = httpx.AsyncClient(timeout=30.0)

    task = asyncio.create_task(
        process_user_input(),
        name="queue-consumer",
    )

    def _on_consumer_done(t: asyncio.Task):
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error("队列消费者异常退出", exc_info=exc)

    task.add_done_callback(_on_consumer_done)

    try:
        yield
    finally:
        # ========== 关闭 ==========
        logger.info(" * 正在关闭 %s ...", app.title)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        if http_client is not None:
            await http_client.aclose()
            http_client = None

        logger.info(" * %s 已关闭", app.title)



app = FastAPI(lifespan=lifespan, title="RAG 多 Agent 聊天机器人")


@app.post("/query")
async def user_query(user_input: UserInput):
    """处理用户查询：入队后立即返回"""
    try:
        await queue.put(user_input)
        logger.info(" * 用户查询已成功加入队列！")
        return {"message": "用户查询已成功加入队列！"}
    except Exception as error:
        logger.exception(" * 入队失败")
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/webhook")
async def receive_callback(data: dict, request: Request):
    logger.info("\n--- 收到回调 ---")
    logger.info("来自: %s:%s", request.client.host, request.client.port)
    logger.info("回调数据: %s\n-------------------------", data)
    return {"status": "success", "message": "回调已接收"}

from pydantic import BaseModel

class SyncResponse(BaseModel):
    query: str
    response: str

@app.post("/query_sync", response_model=SyncResponse)
async def user_query_sync(user_input: UserInput):
    """同步处理：等 agent 跑完再返回"""
    logger.info(" * [sync] 收到查询: %s", user_input.query)
    result = await asyncio.to_thread(
        responder_agent.invoke,
        {"messages": [{"role": "user", "content": user_input.query}]},
    )
    answer = result.get("messages")[-1].content
    return SyncResponse(query=user_input.query, response=answer)

if __name__ == "__main__":
    ...