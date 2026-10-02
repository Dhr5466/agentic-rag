"""Agents module"""

from config import settings
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(
    model="deepseek-flash",
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

if __name__ == "__main__":
    ...