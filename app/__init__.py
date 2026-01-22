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
    """User input"""

    query: str
    user_id: str

    @field_validator("query")
    def validate_query_has_letters(cls, value):
        """Validate that query contains at least one letter"""
        if not re.search(r"[a-zA-Z]", value):
            raise ValueError("Query must contain words")
        return value


async def enqueue_query(user_input: UserInput) -> None:
    """Enqueue user input"""
    await queue.put(user_input)


async def process_user_input() -> dict:
    """Process user input"""
    while True:
        user_input = await queue.get()
        logger.info(" * Processing user input: %s", user_input)
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
                logger.info(" * WEBHOOK RESPONSE: %s", webhook_response)
            except Exception as error:
                logger.error(error)
        logger.info(" * RAG RESPONSE: %s", query_response)
        queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI app lifespan"""
    logger.info(" * %s app started", app.title)
    logger.info(" * Indexing documents in RAG pipeline")
    utils.index_documents()
    task = asyncio.create_task(process_user_input())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, title="RAG multiagent chatbot")


@app.post("/query")
async def user_query(user_input: UserInput, background_tasks: BackgroundTasks):
    """Handle user query"""
    # Once the RAG response has been generated, send to a webhook
    try:
        background_tasks.add_task(enqueue_query, user_input)
        logger.info(" * User query enqueued successfully!")
        return {"message": "User query enqueued successfully!"}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/webhook")
async def receive_callback(data: dict, request: Request):
    logger.info("\n--- RECEIVED CALLBACK ---")
    logger.info("From: %s:%s", request.client.host, request.client.port)
    logger.info("Callback Data: %s\n-------------------------", data)
    return {"status": "success", "message": "Callback received"}


if __name__ == "__main__":
    ...
