"""eval_direct.py — 批量评估，直接调 agent"""
from rag import RAGIngestion
from rag.vector import vector_store
from agents.responder import responder_agent
import time
ids = vector_store.get().get("ids", [])
if not ids:
    print("[eval] 向量库为空，开始索引...")
    t0 = time.time()
    RAGIngestion(docs_directory="documents")
    print(f"[eval] 索引完成，耗时 {time.time() - t0:.1f}s")
else:
    print(f"[eval] 向量库已有 {len(ids)} 条，跳过索引")



def ask(query: str) -> str:
    result = responder_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


TESTS = [
    ("这篇文章讲了什么",             ["Abstract", "Introduction"]),
    ("作者用了哪些 PDF 解析器",      ["4.3 PDF Parsing"]),
    ("RQ1 的结论是什么",             ["Experimental Results"]),
    ("pdfminer 和 pdfplumber 哪个好", ["4.3 PDF Parsing", "Experimental Results"]),
    ("chunk overlap 设多少最好",     ["4.4 Text Chunking", "Experimental Results"]),
    ("GPT-5 在 TableQuest 上准确率", ["Experimental Results"]),
    ("这篇文章有什么局限",           ["Threats to Validity"]),
    ("TableQuest 有多少 QA",         ["TableQuest"]),
    ("六个分块策略都是什么",         ["4.4 Text Chunking"]),
    ("论文用了哪些检索器",           ["4.5 Experimented Embedding Models"]),
]


def check(answer: str, expected: list) -> bool:
    text = answer.lower()
    for exp in expected:
        key = " ".join(exp.split()[-2:]).lower()   # 取后两个词做关键词
        if key in text:
            return True
    return False


hits = 0
t_total = 0.0

for i, (q, expected) in enumerate(TESTS, 1):
    print("=" * 70)
    print(f"[{i}/{len(TESTS)}] {q}")
    t0 = time.time()
    try:
        answer = ask(q)
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}\n")
        continue
    dt = time.time() - t0
    t_total += dt

    ok = check(answer, expected)
    hits += int(ok)
    print(f"  {'✅' if ok else '❌'} 命中={ok}  耗时={dt:.1f}s")
    print(f"  前 150 字: {answer[:150].replace(chr(10), ' ')}")
    print()

print("=" * 70)
print(f"总命中: {hits}/{len(TESTS)} ({hits/len(TESTS)*100:.1f}%)")
print(f"总耗时: {t_total:.1f}s  平均 {t_total/len(TESTS):.1f}s/条")