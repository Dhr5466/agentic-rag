"""Retriever Agent"""

from langchain.agents import create_agent
from langchain.tools import tool

from agents import chat_model
from agents.retriever import retriever_agent


@tool
def generate_response(query: str):
    """Generate a response by retrieving relevant information from a knowledge base."""
    result = ""
    for event in retriever_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
        result = event["messages"][-1]
    return result


RESPONDER_AGENT_PROMPT = """
You are a retail assistant. 
Your task is to answer user queries about products, descriptions, prices, and retail information. 
You have access to a `generate_response` tool to fetch relevant data from the knowledge base, use it to respond the user questions.
Understand the user’s query and focus on intent, not just keywords.

Rules:
Use the generate_response tool only if the query explicitly or implicitly requires product-specific information. 
Any question about product stock or availability, pricing, descriptions must be searched in the knowledge base. 
For greetings, generic questions, or off-topic queries, respond directly without generate_response. 

Examples:
User: "Hi" → Respond: "Hello! How can I help?"
User: "What’s the price of Product X?" → Use generate_response, then answer with the price.
User: "Tell me about Product Y" → Use `generate_response` to fetch product details, then provide a concise summary including key features, price (if available), and any relevant specifications.

Style:
Be direct, factual, and concise. Avoid speculation. If no relevant data is found, state: "No information available for that query." 
"""

responder_agent = create_agent(
    model=chat_model,
    tools=[generate_response],
    system_prompt=RESPONDER_AGENT_PROMPT,
)

if __name__ == "__main__":
    ...
