"""Retriever Agent：执行语义搜索"""

import logging

from langchain.agents import create_agent
from langchain.tools import tool

from agents import chat_model
from config import settings
from rag import RAGIngestion
from rag.vector import vector_store

logger = logging.getLogger(__name__)


@tool
def retrieve_context(query: str):
    """通过在向量数据库中进行语义搜索来检索相关文档。"""
    retrieved_docs = vector_store.similarity_search(query, k=settings.TOP_K)
    serialized = "\n\n".join(
        f"来源: {doc.metadata}\n内容: {doc.page_content}\n元数据: {doc.metadata}"
        for doc in retrieved_docs
    )
    logger.info("已检索文档: %s", len(retrieved_docs))
    return serialized, retrieved_docs


RETRIEVER_AGENT_PROMPT = """
你是一个可以访问上下文检索工具的代理。
只返回最相关的文档块，并按相关性排序。
清晰地格式化结果：文档标题、相关块和引用（如果可用）。
如果不存在相关结果，请说明：'未找到匹配的文档。'
不要在知识库之外生成答案。
保持回复直接且简洁。
"""

retriever_agent = create_agent(
    model=chat_model,
    tools=[retrieve_context],
    system_prompt=RETRIEVER_AGENT_PROMPT,
)

if __name__ == "__main__":
    ...
