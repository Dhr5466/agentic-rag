"""RAG Pipeline"""

import os
from logging import getLogger

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = getLogger(__name__)


class RAGIngestionException(Exception):
    """RAG Ingestion Exception"""


class RAGIngestion:
    """RAG Ingestion Pipeline"""

    def __init__(self, docs_directory: str):
        """Initialize the pipeline"""
        self.directory: str = docs_directory
        self.docs: list[Document] | list[list[Document]] = []
        self.splits: list[Document] = []
        self.document_ids: list[str] = []
        # TODO: reconfigure logging class to show class name in logs
        logger.info("%s: Initializing RAG pipeline", self.__class__.__name__)
        self.vector_store: VectorStore | Chroma = self.setup_vector_database()
        logger.info("%s: Vector database initialized", self.__class__.__name__)
        self.load_documents()
        logger.info("%s: Loaded %s documents", self.__class__.__name__, len(self.docs))
        self.split_documents()
        logger.info(
            "%s: Documents split into %s sub-documents.",
            self.__class__.__name__,
            len(self.splits),
        )
        self.store_documents()
        logger.info(
            "%s: Sub-documents added to vector store: %s",
            self.__class__.__name__,
            len(self.document_ids),
        )

    @staticmethod
    def setup_vector_database():
        """Create the vector store or restore if it exists"""
        return Chroma(
            collection_name="product_collection",
            embedding_function=MistralAIEmbeddings(api_key=settings.MISTRAL_API_KEY),
            persist_directory=settings.VECTOR_DB_DIRECTORY,  # File based vector db
        )

    def load_documents(self):
        """Load documents from a directory"""
        # TODO: standardize this `docs_directory`
        if not self.directory:
            raise RAGIngestionException(f"{self.__class__.__name__}: No directory loaded")
        self.vector_store.reset_collection()
        docs_directory = os.path.join(os.path.dirname(__file__), "..", self.directory)
        file_list = []
        for filename in os.listdir(os.path.join(docs_directory)):
            if filename.endswith((".txt", ".md")):
                file_list.append(os.path.join(docs_directory, filename))
        self.docs = [TextLoader(file_).load() for file_ in file_list]

    def split_documents(self):
        """Split documents from a directory"""
        # TODO: make chunk parameters env vars
        if not self.docs:
            raise RAGIngestionException(f"{self.__class__.__name__}: No documents loaded")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,  # chunk size (characters)
            chunk_overlap=settings.CHUNK_OVERLAP_SIZE,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        docs_list = [item for sublist in self.docs for item in sublist]
        self.splits = text_splitter.split_documents(docs_list)

    def store_documents(self):
        """Store documents from a directory"""
        self.vector_store.reset_collection()
        if not self.splits:
            raise RAGIngestionException(f"{self.__class__.__name__}: No sub-documents loaded")
        self.document_ids = self.vector_store.add_documents(documents=self.splits)
        logger.info("Indexed documents: %s", len(self.vector_store.get().get("ids")))
