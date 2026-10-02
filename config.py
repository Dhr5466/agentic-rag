"""Config module"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Settings class"""
    model_config = SettingsConfigDict(env_file=".env")

    DEEPSEEK_API_KEY: str = ""
    DOCUMENTS_DIRECTORY: str = "documents"
    VECTOR_DB_DIRECTORY: str = ".vector_db"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP_SIZE: int = 100
    TOP_K: int = 8  #这次检索最多返回多少个最相关的 Document / chunk。
    CALLBACK_URL: str = ""

    # 新增两项
    MODEL_NAME: str = r"D:\models\bge-small-zh-v1.5"
    MODEL_KWARGS: dict = Field(default_factory=lambda: {"device": "cuda"})


settings = Settings()


if __name__ == "__main__":
    ...
