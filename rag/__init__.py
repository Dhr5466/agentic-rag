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

import hashlib
import json
logger = getLogger(__name__)


class RAGIngestionException(Exception):
    """RAG Ingestion Exception"""


class RAGIngestion:
    """RAG Ingestion Pipeline

    - reset=True : 清空整个向量库后全量重建
    - 新文件     : 增量入库
    - 文件修改   : 删除旧 chunk 后重新入库
    - 文件未变   : 跳过
    - 文件删除   : 删除对应 chunk
    - 任何一步失败，不写入哈希文件，下次启动重试
    """

    HASH_FILE_NAME = ".rag_file_hashes.json"

    def __init__(self, docs_directory: str, reset: bool = False):
        self.reset = reset
        self.directory = docs_directory

        # 向量库
        self.vector_store: VectorStore | Chroma = vector_store

        # 文档目录（绝对路径）
        self.docs_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", self.directory)
        )

        # 哈希文件放在文档目录之外，避免随文档一起被拷贝/移动
        self.hash_file = os.path.join(
            os.path.dirname(self.docs_directory), self.HASH_FILE_NAME
        )

        # 状态
        self.docs: list[Document] = []
        self.splits: list[Document] = []
        self.document_ids: list[str] = []

        self.saved_hashes: dict[str, str] = {}
        self.current_hashes: dict[str, str] = {}
        # 统一用「相对路径」当键
        self.changed_files: list[str] = []   # 需要新增/重建的相对路径
        self.removed_files: list[str] = []   # 需要删除的相对路径

        logger.info("%s: 初始化 RAG 管道", self.__class__.__name__)
        logger.info("%s: 文档目录 = %s", self.__class__.__name__, self.docs_directory)
        logger.info("%s: 哈希文件 = %s", self.__class__.__name__, self.hash_file)

        # ---------------------------------------------------------
        # 1. 重置向量库
        # ---------------------------------------------------------
        if self.reset:
            logger.warning("%s: reset=True，清空整个向量库", self.__class__.__name__)
            self._reset_vector_store()

        # ---------------------------------------------------------
        # 2. 读取旧哈希
        # ---------------------------------------------------------
        if not self.reset:
            self.saved_hashes = self.load_hashes()

        # ---------------------------------------------------------
        # 3. 扫描 & 计算当前哈希（键=相对路径）
        # ---------------------------------------------------------
        current_files = self.get_file_list()  # 绝对路径列表
        self.current_hashes = {
            self.get_relative_path(p): self.calculate_file_hash(p)
            for p in current_files
        }

        # ---------------------------------------------------------
        # 4. 计算差异
        # ---------------------------------------------------------
        if self.reset:
            self.changed_files = list(self.current_hashes.keys())
        else:
            for rel_path, cur_hash in self.current_hashes.items():
                if self.saved_hashes.get(rel_path) != cur_hash:
                    self.changed_files.append(rel_path)

            self.removed_files = [
                rel for rel in self.saved_hashes
                if rel not in self.current_hashes
            ]

        logger.info(
            "%s: 当前文件 %d 个 | 变更 %d | 删除 %d",
            self.__class__.__name__,
            len(current_files),
            len(self.changed_files),
            len(self.removed_files),
        )

        # ---------------------------------------------------------
        # 5. 没有变化直接返回
        # ---------------------------------------------------------
        if not self.changed_files and not self.removed_files and not self.reset:
            logger.info("%s: 文档没有变化，跳过索引", self.__class__.__name__)
            return

        # ---------------------------------------------------------
        # 6. 加载 + 切分（只处理变更文件）
        # ---------------------------------------------------------
        if self.changed_files:
            self.load_documents(self.changed_files)
            logger.info("%s: 已加载 %d 个 Document",
                        self.__class__.__name__, len(self.docs))
            if self.docs:
                self.split_documents()
                logger.info("%s: 切分为 %d 个子文档",
                            self.__class__.__name__, len(self.splits))

        # ---------------------------------------------------------
        # 7. 写入向量库（先删后加）
        # ---------------------------------------------------------
        self.store_documents()

        # ---------------------------------------------------------
        # 8. 成功后再落盘哈希
        # ---------------------------------------------------------
        self.save_hashes(self.current_hashes)
        logger.info("%s: RAG 索引完成", self.__class__.__name__)

    # =============================================================
    # 文件扫描 / 哈希
    # =============================================================

    def get_file_list(self) -> list[str]:
        """扫描文档目录，返回绝对路径列表（不递归子目录）"""
        if not self.docs_directory:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未配置目录")
        if not os.path.isdir(self.docs_directory):
            raise RAGIngestionException(
                f"{self.__class__.__name__}: 文档目录不存在: {self.docs_directory}"
            )

        exts = (".txt", ".md", ".pdf")
        return [
            os.path.join(self.docs_directory, name)
            for name in sorted(os.listdir(self.docs_directory))
            if name.endswith(exts)
            and os.path.isfile(os.path.join(self.docs_directory, name))
        ]

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_relative_path(self, file_path: str) -> str:
        return os.path.relpath(file_path, self.docs_directory)

    # =============================================================
    # 哈希文件持久化
    # =============================================================

    def load_hashes(self) -> dict[str, str]:
        if not os.path.exists(self.hash_file):
            logger.info("%s: 未找到哈希文件，视为首次索引", self.__class__.__name__)
            return {}
        try:
            with open(self.hash_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as ex:
            logger.warning("%s: 读取哈希文件失败: %s，将重新索引",
                           self.__class__.__name__, ex)
            return {}

    def save_hashes(self, hashes: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(self.hash_file), exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info("%s: 已保存 %d 个文件的哈希",
                    self.__class__.__name__, len(hashes))

    # =============================================================
    # 加载 / 切分
    # =============================================================

    def load_documents(self, relative_paths: list[str]):
        """加载指定（相对路径）文件，统一打上 source_file 元数据"""
        self.docs = []
        for rel_path in relative_paths:
            abs_path = os.path.join(self.docs_directory, rel_path)
            logger.info("%s: 加载 %s", self.__class__.__name__, rel_path)
            try:
                if abs_path.lower().endswith(".pdf"):
                    documents = self.load_pdf(abs_path)
                else:
                    documents = TextLoader(abs_path).load()
            except Exception as ex:
                # 单个文件失败不影响其他文件
                logger.error("%s: 加载 %s 失败: %s",
                             self.__class__.__name__, rel_path, ex)
                raise RAGIngestionException(
                    f"加载文件失败: {rel_path}"
                ) from ex

            for doc in documents:
                doc.metadata["source_file"] = rel_path
            self.docs.extend(documents)

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        pages = pymupdf4llm.to_markdown(
            str(file_path), page_chunks=True, header=False, footer=False,
        )
        documents = []
        for page in pages:
            text = page["text"].strip()
            if not text:
                continue
            documents.append(
                Document(page_content=text, metadata=dict(page["metadata"]))
            )
        return documents

    def split_documents(self):
        if not self.docs:
            raise RAGIngestionException(f"{self.__class__.__name__}: 未加载文档")

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section"), ("###", "subsection")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
            add_start_index=True,
            separators=[
                "\n\n", "\n", "。", "！", "？", "；",
                ".", "!", "?", ";",
                "，", "、", ",", " ", "",
            ],
        )

        first_pass: list[Document] = []
        for doc in self.docs:
            for s in md_splitter.split_text(doc.page_content):
                s.metadata.update(doc.metadata)
                first_pass.append(s)

        self.splits = []
        for s in first_pass:
            if len(s.page_content) <= settings.CHUNK_SIZE:
                self.splits.append(s)
            else:
                self.splits.extend(char_splitter.split_documents([s]))

    # =============================================================
    # 向量库操作
    # =============================================================

    def _reset_vector_store(self):
        """兼容不同 store 的清空方式"""
        try:
            self.vector_store.reset_collection()      # chromadb 原生
            return
        except Exception:
            pass
        try:
            self.vector_store.delete_collection()     # langchain-chroma
            return
        except Exception as ex:
            logger.warning("%s: 无法清空向量库: %s", self.__class__.__name__, ex)

    def delete_file_documents(self, relative_path: str):
        """按 source_file 删除某个文件对应的所有 chunk"""
        logger.info("%s: 删除旧数据: %s", self.__class__.__name__, relative_path)
        try:
            self.vector_store.delete(where={"source_file": relative_path})
        except Exception as ex:
            raise RAGIngestionException(
                f"{self.__class__.__name__}: 删除 {relative_path} 失败: {ex}"
            ) from ex

    def store_documents(self):
        """先删除（已删除的文件 + 发生变化的文件的旧数据），再添加新 chunk"""
        # 1. 删除已不存在的文件
        for rel_path in self.removed_files:
            self.delete_file_documents(rel_path)

        # 2. 删除变更文件的旧数据（reset 时 collection 已空，跳过）
        if not self.reset:
            for rel_path in self.changed_files:
                self.delete_file_documents(rel_path)

        # 3. 添加新 chunk
        if self.splits:
            self.document_ids = self.vector_store.add_documents(
                documents=self.splits
            )
            logger.info("%s: 新增 %d 个 chunk",
                        self.__class__.__name__, len(self.document_ids))
        else:
            logger.info("%s: 没有新 chunk 需要写入", self.__class__.__name__)

        try:
            total = len(self.vector_store.get().get("ids", []) or [])
            logger.info("%s: 当前向量库共 %d 条数据",
                        self.__class__.__name__, total)
        except Exception as ex:
            logger.debug("统计向量库条数失败: %s", ex)


    @staticmethod
    def ensure_indexed(docs_directory: str = "documents", reset: bool = False):
        """启动时调用一次即可"""
        return RAGIngestion(docs_directory=docs_directory, reset=reset)


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