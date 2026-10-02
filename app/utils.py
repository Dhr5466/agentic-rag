"""Utils module"""

import logging

from config import settings
from rag import RAGIngestion

logger = logging.getLogger(__name__)


def index_documents():
    """主流程"""
    logger.info("正在索引 %s", settings.DOCUMENTS_DIRECTORY)
    try:
        RAGIngestion(docs_directory=settings.DOCUMENTS_DIRECTORY)
        logger.info("索引完成！")
    except Exception:
        logger.exception("RAG 索引失败，检索功能可能不可用")
        raise

if __name__ == "__main__":
    index_documents()
