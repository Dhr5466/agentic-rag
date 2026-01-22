"""Utils module"""

import logging

from config import settings
from rag import RAGIngestion

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def index_documents():
    """Main pipeline"""
    logger.info("Indexing %s", settings.DOCUMENTS_DIRECTORY)
    try:
        RAGIngestion(docs_directory=settings.DOCUMENTS_DIRECTORY)
        logger.info("Indexing COMPLETE!")
    except Exception as ex:
        logger.error(ex)


if __name__ == "__main__":
    index_documents()
