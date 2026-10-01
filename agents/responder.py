"""Retriever Agent"""

from langchain.agents import create_agent
from langchain.tools import tool

from agents import chat_model
from agents.retriever import retriever_agent


@tool
def generate_response(query: str):
    """通过从知识库中检索相关信息来生成响应。"""
    print("\n" + "=" * 60)
    print("[Tool] generate_response 被调用")
    print("[Tool] query:", query)
    print("=" * 60)

    result = ""
    for event in retriever_agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
        result = event["messages"][-1]
        from pprint import pprint

        pprint(event, width=120)

    print("\n[Tool] Retriever 执行结束")
    print("[Tool] result:", result)
    return result

RESPONDER_AGENT_PROMPT = """
你是一个技术文档与论文阅读助手。

你的任务是帮助用户理解知识库中的论文、技术文档和其他资料。
你可以使用 `generate_response` 工具，从知识库中检索与用户问题相关的内容。

规则：
1. 当用户询问论文、文章、技术文档中的具体内容时，必须使用 generate_response 工具。
2. 当用户询问论文的主题、摘要、研究内容、方法、实验、结果、结论等内容时，必须使用 generate_response 工具。
3. 当用户询问知识库中是否存在某项信息时，使用 generate_response 工具进行检索。
4. 对于简单问候或与知识库无关的问题，可以直接回答，不需要调用工具。
5. 如果知识库中没有找到相关信息，应明确说明没有找到相关内容，不要编造答案。

例如：
用户：“这篇文章主要研究了什么？”
→ 使用 generate_response

用户：“论文用了什么方法？”
→ 使用 generate_response

用户：“实验结果怎么样？”
→ 使用 generate_response

用户：“你好”
→ 直接回复

风格：
简洁、准确、基于检索到的内容回答。
"""

responder_agent = create_agent(
    model=chat_model,
    tools=[generate_response],
    system_prompt=RESPONDER_AGENT_PROMPT,
)

if __name__ == "__main__":
    ...
