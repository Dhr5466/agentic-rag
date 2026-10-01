"""RAG Pipeline"""

import os
from pathlib import Path

import pymupdf4llm
from logging import getLogger

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter

from config import settings
from rag.vector import vector_store

logger = getLogger(__name__)


class RAGIngestionException(Exception):
    """RAG Ingestion Exception"""


class RAGIngestion:
    """RAG Ingestion Pipeline  RAG 知识库的数据入库流水线"""

    def __init__(self, docs_directory: str, reset: bool = False):
        """初始化Pipeline管道"""
        self.reset: bool = reset
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

    def load_documents(self):#把文件 变成 LangChain Document #现在添加pdf
        """从目录加载文档"""
        if not self.directory:
            raise RAGIngestionException(
                f"{self.__class__.__name__}: 未加载目录"
            )

        #self.vector_store.reset_collection()

        docs_directory = os.path.join(
            os.path.dirname(__file__),
            "..",
            self.directory,
        )

        file_list = []

        for filename in os.listdir(docs_directory):
            if filename.endswith((".txt", ".md", ".pdf")):
                file_list.append(
                    os.path.join(docs_directory, filename)
                )

        self.docs = []

        for file_path in file_list:
            if file_path.endswith(".pdf"):
                documents = self.load_pdf(file_path)
            else:
                documents = TextLoader(file_path).load()

            self.docs.extend(documents)   # ← 改成 extend 防止套娃
        """最后加载出来的是这样的有内容也有来源
        Document(page_content="...",metadata={...})
最后是变成一个这个
self.docs = [
    [Document(...)],
    [Document(...)],
    [Document(...)]
]
    """


    def load_pdf(self,file_path: str | Path) -> list[Document]:
        """
        将 PDF 解析成 LangChain Document。
        一页对应一个 Document。
        """

        pages = pymupdf4llm.to_markdown(
            str(file_path),
            page_chunks=True,
            header=False,
            footer=False,
        )

        documents = []

        for page in pages:
            text = page["text"].strip()

            if not text:
                continue

            metadata = dict(page["metadata"])

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        return documents

    def split_documents(self):
        """两阶段切分：先按 Markdown 标题切，再按长度切"""
        if not self.docs:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载文档")

        # 第一步：Markdown 标题切分（保留 section / subsection）
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "section"),  # PyMuPDF4LLM 把一级标题输出成 ##
                ("###", "subsection"),
            ],
            strip_headers=False,  # 标题文字保留在 chunk 里
        )

        # 第二步：超长的再按字符切
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
            add_start_index=True,
            separators=["\n\n", "\n","。", "！", "？", "；",".", "!", "?", ";","，", "、", ",", " ","",],
        )

        first_pass: list[Document] = []
        for doc in self.docs:
            for s in md_splitter.split_text(doc.page_content):
                # 继承原页的 metadata（page_number、title、author 等）
                s.metadata.update(doc.metadata)
                first_pass.append(s)

        # 第二遍切分
        self.splits = []
        for s in first_pass:
            if len(s.page_content) <= settings.CHUNK_SIZE:
                self.splits.append(s)
            else:
                for sub in char_splitter.split_documents([s]):
                    self.splits.append(sub)

    def ensure_indexed(docs_directory: str = "documents"):
        """只在向量库为空时才建索引"""
        from rag.vector import vector_store

        ids = vector_store.get().get("ids", [])
        if ids:
            print(f"[eval] 向量库已有 {len(ids)} 条，跳过索引")
            return

        print("[eval] 向量库为空，开始索引")
        RAGIngestion(docs_directory=docs_directory, reset=False)
    def store_documents(self):
        """存储文档"""
        #TOOD
        if self.reset:
            self.vector_store.reset_collection()

        self.vector_store.reset_collection()#之后可以不要每次服务启动都重建整个知识库，而是做增量索引。
        if not self.splits:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载子文档")
        self.document_ids = self.vector_store.add_documents(documents=self.splits)
        logger.info("已索引文档: %s", len(self.vector_store.get().get("ids")))



if __name__ == "__main__":
    import logging
    import random
    from collections import Counter

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("RAGIngestion 测试")
    print("=" * 70)

    # ---------------------------------------------------------
    # 0. 跑完整 pipeline
    # ---------------------------------------------------------
    pipeline = RAGIngestion(docs_directory="documents")

    print(f"\n原始 Document 数 : {len(pipeline.docs)}")
    print(f"切分后 chunk 数  : {len(pipeline.splits)}")
    print(f"入库 ID 数       : {len(pipeline.document_ids)}")

    # ---------------------------------------------------------
    # 1. metadata 抽样
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. metadata 抽样（前 3 个 chunk）")
    print("=" * 70)
    for i, s in enumerate(pipeline.splits[:3]):
        # source 太长了，截断显示
        meta = {
            k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
            for k, v in s.metadata.items()
        }
        print(f"\n[Chunk {i}]")
        print(f"  metadata: {meta}")
        print(f"  前 150 字: {s.page_content[:150].replace(chr(10), ' ')}")

    # ---------------------------------------------------------
    # 2. 页眉污染检查
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. 页眉污染检查")
    print("=" * 70)
    header_markers = [
        "ICSE-SEIP ’26",
        "ICSE-SEIP '26",
        "April 12–18, 2026",
        "El Bachyr, et al.",
    ]
    polluted = []
    for i, s in enumerate(pipeline.splits):
        for marker in header_markers:
            if marker in s.page_content:
                polluted.append((i, marker))
                break

    if polluted:
        print(f"⚠️  {len(polluted)} 个 chunk 含有页眉特征串：")
        for i, marker in polluted[:5]:
            print(f"   chunk {i}: 命中 '{marker}'")
    else:
        print("✅ 没有发现页眉污染")

    # ---------------------------------------------------------
    # 3. chunk 长度分布
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. chunk 长度分布")
    print("=" * 70)
    lengths = [len(s.page_content) for s in pipeline.splits]
    print(f"  最短 : {min(lengths)}")
    print(f"  最长 : {max(lengths)}")
    print(f"  平均 : {sum(lengths) / len(lengths):.0f}")

    buckets = Counter((l // 200) * 200 for l in lengths)
    print("\n  长度直方图（每 200 字符一档）：")
    for bucket in sorted(buckets):
        bar = "█" * buckets[bucket]
        print(f"    {bucket:>5}-{bucket + 199:<5} {bar} {buckets[bucket]}")

    # ---------------------------------------------------------
    # 4. 随机抽 5 个 chunk 看内容
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. 随机抽 5 个 chunk")
    print("=" * 70)
    random.seed(42)
    sample = random.sample(pipeline.splits, min(5, len(pipeline.splits)))
    for i, s in enumerate(sample):
        print(f"\n--- 抽样 {i + 1} ---")
        print(f"  metadata: {s.metadata}")
        print(f"  {s.page_content[:250]}")
        print("  ...")

    # ---------------------------------------------------------
    # 5. 入库结果核对
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. 入库结果核对")
    print("=" * 70)
    stored = pipeline.vector_store.get()
    stored_ids = stored.get("ids", [])
    print(f"  vector_store 里实际 ID 数: {len(stored_ids)}")
    print(f"  pipeline.document_ids   : {len(pipeline.document_ids)}")
    if len(stored_ids) == len(pipeline.document_ids):
        print("  ✅ 一致")
    else:
        print("  ⚠️ 不一致（可能是 reset_collection 被调用了两次）")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)