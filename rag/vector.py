"""Vector database module 向量数据库模块"""

from langchain_chroma import Chroma
from config import settings
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name=r"D:\models\bge-small-zh-v1.5",  # 用 BAAI/bge-small-zh-v1.5 这个模型做文本向量化
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # 这个是做向量归一化，让向量长度变成 1
)#把一句话转换成一串数字（向量）的工具。

vector_store = Chroma(
    collection_name="product_collection",
    embedding_function=embedding_model,  # 以后 Chroma 收到文本的时候，就自动调用这个 Embedding 模型，把文本转成向量。
    persist_directory=settings.VECTOR_DB_DIRECTORY,  # 基于文件的向量数据库
)#专门帮你保存这些向量，并且根据“相似程度”把相关内容找出来的向量数据库。


if __name__ == "__main__":
    from langchain_core.documents import Document
    print("=" * 60)
    print("1. 测试 Embedding")
    print("=" * 60)

    # ---------- 输入 ----------
    text = "这个商品还有库存吗？"

    print("输入文本：")
    print(text)

    # ---------- 文本 -> 向量 ----------
    vector = embedding_model.embed_query(text)

    print("\n输出向量：")
    print(vector[:10], "...")
    print("向量长度：", len(vector))

    print("\n" + "=" * 60)
    print("2. 测试多条文本 Embedding")
    print("=" * 60)

    texts = [
        "这个商品还有库存吗？",
        "现在还有货吗？",
        "这个商品多少钱？",
        "今天天气怎么样？",
    ]

    vectors = embedding_model.embed_documents(texts)

    for i, (text, vector) in enumerate(zip(texts, vectors)):
        print(f"\n文本 {i + 1}: {text}")
        print(f"向量前10维: {vector[:10]}")
        print(f"向量长度: {len(vector)}")

    print("\n" + "=" * 60)
    print("3. 把文本存进 Chroma")
    print("=" * 60)

    documents = [
        Document(
            page_content="Mystery Book 当前库存为 0，本商品暂时无货。",
            metadata={"source": "stock.txt"},
        ),
        Document(
            page_content="Mystery Book 的价格是 111 美元。",
            metadata={"source": "prices.txt"},
        ),
        Document(
            page_content="Mystery Book 的产品编号是 1427b3b0。",
            metadata={"source": "product_ids.txt"},
        ),
        Document(
            page_content="今天天气很好，适合出去散步。",
            metadata={"source": "weather.txt"},
        ),
    ]

    # 存入 Chroma
    ids = vector_store.add_documents(documents)

    print("成功存入文档数量：", len(ids))
    print("文档 ID：", ids)

    print("\n" + "=" * 60)
    print("4. 从 Chroma 检索")
    print("=" * 60)

    # ---------- 输入问题 ----------
    query = "Mystery Book 还有货吗？"

    print("查询问题：")
    print(query)

    # ---------- 相似度检索 ----------
    results = vector_store.similarity_search_with_score(
        query,
        k=3,
    )

    print("\n检索结果：")

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print("内容：", doc.page_content)
        print("来源：", doc.metadata)
        print("相似度/距离分数：", score)
"""这个 score 在这里不是“越大越相似”Chroma 的这个 similarity_search_with_score() 返回的是距离分数，在你这个设置下可以理解成：距离越小 → 越相似。"""