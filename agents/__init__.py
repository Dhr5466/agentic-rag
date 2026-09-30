"""Agents module"""

from langchain_mistralai import ChatMistralAI

from config import settings
from langchain_openai import ChatOpenAI
#chat_model = ChatMistralAI(model_name="mistral-large-latest", api_key=settings.MISTRAL_API_KEY)
chat_model = ChatOpenAI(
    model="deepseek-flash",
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

if __name__ == "__main__":
    ...