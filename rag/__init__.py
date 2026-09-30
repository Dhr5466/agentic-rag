"""RAG Pipeline"""

import os
from logging import getLogger

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.vector import vector_store

logger = getLogger(__name__)


class RAGIngestionException(Exception):
    """RAG Ingestion Exception"""


class RAGIngestion:
    """RAG Ingestion Pipeline  RAG 知识库的数据入库流水线"""

    def __init__(self, docs_directory: str):
        """初始化Pipeline管道"""
        self.directory: str = docs_directory#文档目录
        self.docs: list[Document] | list[list[Document]] = []#原始文档
        self.splits: list[Document] = []#切完之后的小块
        self.document_ids: list[str] = []#存进向量数据库后的 ID
        self.vector_store: VectorStore | Chroma = vector_store
        # TODO: 重新配置日志类以在日志中显示类名
        logger.info("%s: 正在初始化 RAG 管道", self.__class__.__name__)
        self.load_documents()
        logger.info("%s: 已加载 %s 个文档", self.__class__.__name__, len(self.docs))
        self.split_documents()
        logger.info(
            "%s: 文档已拆分为 %s 个子文档。",
            self.__class__.__name__,
            len(self.splits),
        )
        self.store_documents()
        logger.info(
            "%s: 子文档已添加到向量存储: %s",
            self.__class__.__name__,
            len(self.document_ids),
        )

    def load_documents(self):#把文件 变成 LangChain Document
        """从目录加载文档"""
        # TODO: 标准化此 `docs_directory`
        if not self.directory:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载目录")
        self.vector_store.reset_collection()
        docs_directory = os.path.join(os.path.dirname(__file__), "..", self.directory)
        file_list = []
        for filename in os.listdir(os.path.join(docs_directory)):
            if filename.endswith((".txt", ".md")):
                file_list.append(os.path.join(docs_directory, filename))
        self.docs = [TextLoader(file_).load() for file_ in file_list]
        """最后加载出来的是这样的有内容也有来源
        Document(page_content="...",metadata={...})
最后是变成一个这个
self.docs = [
    [Document(...)],
    [Document(...)],
    [Document(...)]
]
    """

    def split_documents(self):#大 Document 变成 小 Document
        """从目录拆分文档"""
        # TODO: 将块参数设为环境变量
        if not self.docs:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载文档")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,  # 块大小（字符数）
            chunk_overlap=settings.CHUNK_OVERLAP_SIZE,  # 块重叠（字符数）为了将语序连贯
            add_start_index=True,  # 跟踪原始文档中的索引 记录这个 chunk 在原始文本中从什么位置开始。
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                ".",
                "!",
                "?",
                ";",
                "，",
                "、",
                ",",
                " ",
                ""
            ],#因为这里默认是英语的文本得修改为中文也可以的
        )#这叫递归字符切分器  尽量按照比较自然的边界，把长文本切成指定长度的小块
        docs_list = [item for sublist in self.docs for item in sublist] #原版这里是列表里面套列表，这一句就是把它摊平
        self.splits = text_splitter.split_documents(docs_list)

    def store_documents(self):
        """存储文档"""
        self.vector_store.reset_collection()#之后可以不要每次服务启动都重建整个知识库，而是做增量索引。
        if not self.splits:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载子文档")
        self.document_ids = self.vector_store.add_documents(documents=self.splits)
        logger.info("已索引文档: %s", len(self.vector_store.get().get("ids")))


if __name__ == "__main__":
    ...
