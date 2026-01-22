"""Config module"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings class"""

    MISTRAL_API_KEY: str = ""
    DOCUMENTS_DIRECTORY: str = "documents"
    VECTOR_DB_DIRECTORY: str = ".vector_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP_SIZE: int = 100
    TOP_K: int = 5

    class Config:
        """Config class"""

        env_file = ".env"


settings = Settings()


if __name__ == "__main__":
    ...
