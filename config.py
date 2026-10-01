"""Config module"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings class"""
    model_config = SettingsConfigDict(env_file=".env")

    DEEPSEEK_API_KEY: str = ""
    DOCUMENTS_DIRECTORY: str = "documents"
    VECTOR_DB_DIRECTORY: str = ".vector_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP_SIZE: int = 100
    TOP_K: int = 5
    CALLBACK_URL: str = ""


settings = Settings()


if __name__ == "__main__":
    ...
