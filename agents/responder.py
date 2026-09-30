"""Retriever Agent"""

from langchain.agents import create_agent
from langchain.tools import tool

from agents import chat_model
from agents.retriever import retriever_agent


@tool
def generate_response(query: str):
    """通过从知识库中检索相关信息来生成响应。"""
    result = ""
    for event in retriever_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
        result = event["messages"][-1]
        from pprint import pprint

        pprint(event, width=120)
    return result

RESPONDER_AGENT_PROMPT = """
你是一个零售助手。
你的任务是回答用户关于产品、描述、价格和零售信息的查询。
你可以访问一个 `generate_response` 工具，从知识库中获取相关数据，并用它来回答用户的问题。
理解用户的查询，关注意图，而不仅仅是关键词。

规则：
仅当查询明确或隐含地需要产品特定信息时，才使用 generate_response 工具。
任何关于产品库存或可用性、定价、描述的问题都必须在知识库中搜索。
对于问候、一般性问题或无关查询，直接回复，不要使用 generate_response。

示例：
用户："Hi" → 回复："Hello! How can I help?"
用户："What’s the price of Product X?" → 使用 generate_response，然后用价格回答。
用户："Tell me about Product Y" → 使用 `generate_response` 获取产品详情，然后提供一个简洁的摘要，包括关键特性、价格（如果可用）以及任何相关规格。

风格：
直接、基于事实、简洁。避免推测。如果未找到相关数据，请说明："No information available for that query."
"""

responder_agent = create_agent(
    model=chat_model,
    tools=[generate_response],
    system_prompt=RESPONDER_AGENT_PROMPT,
)

if __name__ == "__main__":
    ...
