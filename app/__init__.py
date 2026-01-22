"""FastAPI main app"""

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(debut=True, title="RAG multiagent chatbot")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserInput(BaseModel):
    """User input"""

    query: str
    user_id: str


async def process_user_input(user_input) -> dict:
    """Process user input"""
    result = {"query_size": len(user_input.query), "user_id": user_input.user_id}
    logger.info("Result: %s", result)
    return result


@app.post("/query")
async def user_query(user_input: UserInput, background_tasks: BackgroundTasks):
    """Handle user query"""
    # Validate input and enqueue the request
    # In the background, send the query to the RAG
    # Once the RAG response has been generated, send to a webhook
    try:
        background_tasks.add_task(process_user_input, user_input)
        return {"message": "User query waiting to be processed."}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
