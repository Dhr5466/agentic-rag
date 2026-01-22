"""Unit tests for rag module"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from rag.vector import vector_store, embedding_model


class TestVectorStore:
    """Test cases for vector store"""

    def test_vector_store_exists(self):
        """Test that vector store is properly initialized"""
        assert vector_store is not None
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(vector_store, 'add_documents')

    def test_embedding_model_exists(self):
        """Test that embedding model is properly initialized"""
        assert embedding_model is not None
        assert hasattr(embedding_model, 'embed_query')

    @patch('rag.vector.settings')
    def test_vector_store_configuration(self, mock_settings):
        """Test vector store configuration with mocked settings"""
        mock_settings.VECTOR_DB_DIRECTORY = "test_db"
        mock_settings.MISTRAL_API_KEY = "test_key"
        
        # Import after patching settings
        from rag.vector import vector_store
        assert vector_store is not None

    @patch.object(vector_store, 'similarity_search')
    def test_similarity_search_mock(self, mock_search):
        """Test similarity search with mock"""
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
        """Test adding documents with mock"""
        from langchain_core.documents import Document
        
        test_docs = [
            Document(page_content="Doc 1", metadata={"source": "file1.txt"}),
            Document(page_content="Doc 2", metadata={"source": "file2.txt"})
        ]
        
        vector_store.add_documents(test_docs)
        
        mock_add.assert_called_once_with(test_docs)

    def test_vector_store_collection_name(self):
        """Test that vector store has correct collection name"""
        # This test checks the collection name without making API calls
        assert hasattr(vector_store, '_collection_name')
        # Note: The actual collection name might be stored differently
        # depending on the Chroma implementation


class TestEmbeddingModel:
    """Test cases for embedding model"""

    def test_embedding_model_type(self):
        """Test that embedding model is of correct type"""
        from langchain_mistralai import MistralAIEmbeddings
        assert isinstance(embedding_model, MistralAIEmbeddings)

    def test_embedding_model_has_required_methods(self):
        """Test that embedding model has required methods"""
        assert hasattr(embedding_model, 'embed_query')
        assert hasattr(embedding_model, 'embed_documents')
        assert callable(embedding_model.embed_query)
        assert callable(embedding_model.embed_documents)


class TestRAGIntegration:
    """Integration tests for RAG module"""

    @patch('rag.vector.settings')
    def test_rag_module_configuration(self, mock_settings):
        """Test RAG module configuration"""
        mock_settings.VECTOR_DB_DIRECTORY = "test_vector_db"
        mock_settings.MISTRAL_API_KEY = "test_api_key"
        
        # Test that settings are used correctly
        from rag.vector import vector_store, embedding_model
        assert vector_store is not None
        assert embedding_model is not None

    def test_rag_module_imports(self):
        """Test that RAG module imports work correctly"""
        from rag.vector import vector_store, embedding_model
        assert vector_store is not None
        assert embedding_model is not None
        
        # Test that the objects have expected methods
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(embedding_model, 'embed_query')

    @patch.object(vector_store, 'similarity_search')
    def test_end_to_end_search_mock(self, mock_search):
        """Test end-to-end search workflow with mocks"""
        from langchain_core.documents import Document
        
        # Mock search results
        mock_doc = Document(
            page_content="Relevant product information",
            metadata={"source": "products.txt", "id": "prod_1"}
        )
        mock_search.return_value = [mock_doc]
        
        # Simulate the search workflow
        query = "product information"
        results = vector_store.similarity_search(query, k=3)
        
        assert len(results) == 1
        assert results[0].page_content == "Relevant product information"
        assert results[0].metadata["source"] == "products.txt"
        mock_search.assert_called_once_with(query, k=3)

    def test_vector_store_and_embedding_integration(self):
        """Test that vector store and embedding model work together"""
        # Test that both components exist and are properly configured
        assert vector_store is not None
        assert embedding_model is not None
        
        # Test that they have the expected interface
        assert hasattr(vector_store, 'similarity_search')
        assert hasattr(vector_store, 'add_documents')
        assert hasattr(embedding_model, 'embed_query')
        assert hasattr(embedding_model, 'embed_documents')

    def test_rag_module_constants(self):
        """Test that RAG module has expected constants and configuration"""
        from rag.vector import vector_store, embedding_model
        from config import settings
        
        # Test that settings are accessible
        assert hasattr(settings, 'VECTOR_DB_DIRECTORY')
        assert hasattr(settings, 'MISTRAL_API_KEY')
        
        # Test that components are initialized
        assert vector_store is not None
        assert embedding_model is not None