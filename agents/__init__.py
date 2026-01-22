"""Agents module"""

from langchain_mistralai import ChatMistralAI

from config import settings

chat_model = ChatMistralAI(model_name="mistral-large-latest", api_key=settings.MISTRAL_API_KEY)


if __name__ == "__main__":
    ...