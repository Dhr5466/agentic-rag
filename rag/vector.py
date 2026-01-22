"""Vector database module"""

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

from config import settings

embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=settings.MISTRAL_API_KEY)

vector_store = Chroma(
    collection_name="product_collection",
    embedding_function=embedding_model,
    persist_directory=settings.VECTOR_DB_DIRECTORY,  # File based vector db
)


if __name__ == "__main__":
    ...
