"""ask_direct.py — 直接调用 agent，不走 HTTP / 队列 / 回调"""

import sys
from agents.responder import responder_agent


def ask(query: str) -> str:
    result = responder_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


def main():
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
        return

    print("交互模式（exit 退出）")
    while True:
        try:
            q = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        try:
            print("\n" + ask(q))
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()