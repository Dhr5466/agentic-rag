"""RAG ingestion 模块的单元测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from rag import RAGIngestion, RAGIngestionException


class TestRAGIngestion:
    """RAG ingestion 管道的测试用例"""

    @patch('rag.vector_store')
    @patch('os.listdir')
    @patch('os.path.join')
    def test_rag_ingestion_initialization(self, mock_join, mock_listdir, mock_vector_store):
        """使用 mock 测试 RAG ingestion 初始化"""
        # mock 文件系统
        mock_listdir.return_value = ['doc1.txt', 'doc2.md', 'ignore.pdf']
        mock_join.side_effect = lambda *args: '/'.join(args)

        # mock 向量存储
        mock_vector_store.reset_collection = Mock()
        mock_vector_store.add_documents = Mock(return_value=['id1', 'id2'])
        mock_vector_store.get = Mock(return_value={'ids': ['id1', 'id2']})

        # mock 文档加载和拆分
        with patch('rag.PyPDFLoader') as mock_pdf_loader, \
                patch('rag.TextLoader') as mock_loader, \
                patch('rag.RecursiveCharacterTextSplitter') as mock_splitter:
            # mock 文档加载器
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.txt"}
            mock_loader.return_value.load.return_value = [mock_doc]
            mock_pdf_loader.return_value.load.return_value = [mock_doc]

            # mock 文本拆分器
            mock_splitter_instance = Mock()
            mock_splitter_instance.split_documents.return_value = [mock_doc]
            mock_splitter.return_value = mock_splitter_instance

            # 初始化 RAG ingestion
            rag = RAGIngestion("test_docs")

            # 验证初始化
            assert rag.directory == "test_docs"
            assert len(rag.docs) == 3  #  现在处理 txt + md + pdf 三种
            assert len(rag.splits) == 1
            assert len(rag.document_ids) == 2

    def test_rag_ingestion_exception(self):
        """测试 RAG ingestion 异常"""
        exception = RAGIngestionException("Test error")
        assert str(exception) == "Test error"
        assert isinstance(exception, Exception)

    @patch('rag.vector_store')
    def test_load_documents_no_directory(self, mock_vector_store):
        """测试没有目录时的 load_documents"""
        mock_vector_store.reset_collection = Mock()

        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.directory = ""
            rag.vector_store = mock_vector_store

            with pytest.raises(RAGIngestionException):
                rag.load_documents()

    @patch('rag.vector_store')
    def test_split_documents_no_docs(self, mock_vector_store):
        """测试没有加载文档时的 split_documents"""
        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.docs = []

            with pytest.raises(RAGIngestionException):
                rag.split_documents()

    @patch('rag.vector_store')
    def test_store_documents_no_splits(self, mock_vector_store):
        """测试没有拆分结果时的 store_documents"""
        mock_vector_store.reset_collection = Mock()

        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.splits = []
            rag.vector_store = mock_vector_store

            with pytest.raises(RAGIngestionException):
                rag.store_documents()


class TestRAGIngestionIntegration:
    """RAG ingestion 的集成测试"""

    def test_rag_ingestion_import(self):
        """测试 RAG ingestion 是否可以被导入"""
        from rag import RAGIngestion, RAGIngestionException
        assert RAGIngestion is not None
        assert RAGIngestionException is not None

    @patch('rag.vector_store')
    @patch('os.path.exists')
    def test_rag_ingestion_with_settings(self, mock_exists, mock_vector_store):
        """测试 RAG ingestion 是否正确使用 settings"""
        from config import settings

        # mock 向量存储
        mock_vector_store.reset_collection = Mock()
        mock_vector_store.add_documents = Mock(return_value=[])
        mock_vector_store.get = Mock(return_value={'ids': []})

        # mock 文件系统
        mock_exists.return_value = True

        with patch('os.listdir', return_value=['test.txt']), \
             patch('rag.TextLoader') as mock_loader, \
             patch('rag.RecursiveCharacterTextSplitter') as mock_splitter:

            # mock 文档加载器
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.txt"}
            mock_loader.return_value.load.return_value = [mock_doc]

            mock_splitter_instance = Mock()
            mock_splitter_instance.split_documents.return_value = [mock_doc]
            mock_splitter.return_value = mock_splitter_instance

            # 测试是否使用了 settings
            rag = RAGIngestion("test_docs")

            # 验证文本拆分器是否使用了 settings
            mock_splitter.assert_called_once_with(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
                add_start_index=True,
                separators=['\n\n', '\n', '。', '！', '？', '；', '.', '!', '?', ';', '，', '、', ',', ' ', '']
            )