"""Retriever Agent: perform semantic search"""

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
    """Retrieve relevant documents by doing semantic search in a vector database."""
    retrieved_docs = vector_store.similarity_search(query, k=settings.TOP_K)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}\nMetadata: {doc.metadata}"
        for doc in retrieved_docs
    )
    logger.info("Retrieved documents: %s", len(retrieved_docs))
    return serialized, retrieved_docs


RETRIEVER_AGENT_PROMPT = """
You are an agent with access to a context retriever tool.  
Return only the most relevant document chunks, ranked by relevance. 
Format results clearly: document title, relevant chunk, and reference (if available). 
If no relevant results exist, state: 'No matching documents found.' 
Do not generate answers outside the knowledge base. 
Keep responses direct and minimal. 
"""

retriever_agent = create_agent(
    model=chat_model,
    tools=[retrieve_context],
    system_prompt=RETRIEVER_AGENT_PROMPT,
)

if __name__ == "__main__":
    ...
