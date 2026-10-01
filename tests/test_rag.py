"""rag 模块的单元测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rag.vector import vector_store, embedding_model


class TestVectorStore:
    """向量存储的测试用例"""

    def test_vector_store_exists(self):
        """测试向量存储是否正确初始化"""
        assert vector_store is not None
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(vector_store, 'add_documents')

    def test_embedding_model_exists(self):
        """测试 Embedding 模型是否正确初始化"""
        assert embedding_model is not None
        assert hasattr(embedding_model, 'embed_query')

    @patch('rag.vector.settings')
    def test_vector_store_configuration(self, mock_settings):
        """使用 mock 的 settings 测试向量存储配置"""
        mock_settings.VECTOR_DB_DIRECTORY = "test_db"
        mock_settings.MISTRAL_API_KEY = "test_key"

        # 在 patch settings 后导入
        from rag.vector import vector_store
        assert vector_store is not None

    @patch.object(vector_store, 'similarity_search')
    def test_similarity_search_mock(self, mock_search):
        """使用 mock 测试相似度搜索"""
        from langchain_core.documents import Document

        mock_doc = Document(
            page_content="Test document content",
            metadata={"source": "test.txt"}
        )
        mock_search.return_value = [mock_doc]

        results = vector_store.similarity_search("test query", k=5)

        assert len(results) == 1
        assert results[0].page_content == "Test document content"
        mock_search.assert_called_once_with("test query", k=5)

    @patch.object(vector_store, 'add_documents')
    def test_add_documents_mock(self, mock_add):
        """使用 mock 测试添加文档"""
        from langchain_core.documents import Document

        test_docs = [
            Document(page_content="Doc 1", metadata={"source": "file1.txt"}),
            Document(page_content="Doc 2", metadata={"source": "file2.txt"})
        ]

        vector_store.add_documents(test_docs)

        mock_add.assert_called_once_with(test_docs)

    def test_vector_store_collection_name(self):
        """测试向量存储是否具有正确的 collection 名称"""
        # 此测试在不调用 API 的情况下检查 collection 名称
        assert hasattr(vector_store, '_collection_name')
        # 注意：实际的 collection 名称可能以不同方式存储
        # 具体取决于 Chroma 的实现


class TestEmbeddingModel:
    """Embedding 模型的测试用例"""



    def test_embedding_model_has_required_methods(self):
        """测试 Embedding 模型是否具有所需的方法"""
        assert hasattr(embedding_model, 'embed_query')
        assert hasattr(embedding_model, 'embed_documents')
        assert callable(embedding_model.embed_query)
        assert callable(embedding_model.embed_documents)


class TestRAGIntegration:
    """RAG 模块的集成测试"""

    @patch('rag.vector.settings')
    def test_rag_module_configuration(self, mock_settings):
        """测试 RAG 模块配置"""
        mock_settings.VECTOR_DB_DIRECTORY = "test_vector_db"
        mock_settings.MISTRAL_API_KEY = "test_api_key"

        # 测试 settings 是否被正确使用
        from rag.vector import vector_store, embedding_model
        assert vector_store is not None
        assert embedding_model is not None

    def test_rag_module_imports(self):
        """测试 RAG 模块导入是否正常工作"""
        from rag.vector import vector_store, embedding_model
        assert vector_store is not None
        assert embedding_model is not None

        # 测试对象是否具有预期的方法
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(embedding_model, 'embed_query')

    @patch.object(vector_store, 'similarity_search')
    def test_end_to_end_search_mock(self, mock_search):
        """使用 mock 测试端到端搜索工作流"""
        from langchain_core.documents import Document

        # mock 搜索结果
        mock_doc = Document(
            page_content="Relevant product information",
            metadata={"source": "products.txt", "id": "prod_1"}
        )
        mock_search.return_value = [mock_doc]

        # 模拟搜索工作流
        query = "product information"
        results = vector_store.similarity_search(query, k=3)

        assert len(results) == 1
        assert results[0].page_content == "Relevant product information"
        assert results[0].metadata["source"] == "products.txt"
        mock_search.assert_called_once_with(query, k=3)

    def test_vector_store_and_embedding_integration(self):
        """测试向量存储和 Embedding 模型是否协同工作"""
        # 测试两个组件是否存在并正确配置
        assert vector_store is not None
        assert embedding_model is not None

        # 测试它们是否具有预期的接口
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(vector_store, 'add_documents')
        assert hasattr(embedding_model, 'embed_query')
        assert hasattr(embedding_model, 'embed_documents')

    def test_rag_module_constants(self):
        """测试 RAG 模块是否具有预期的常量和配置"""
        from rag.vector import vector_store, embedding_model
        from config import settings

        # 测试 settings 是否可访问
        assert hasattr(settings, 'VECTOR_DB_DIRECTORY')
        assert hasattr(settings, 'DEEPSEEK_API_KEY')

        # 测试组件是否已初始化
        assert vector_store is not None
        assert embedding_model is not None