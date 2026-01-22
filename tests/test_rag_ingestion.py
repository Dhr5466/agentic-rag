"""Unit tests for RAG ingestion module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from rag import RAGIngestion, RAGIngestionException


class TestRAGIngestion:
    """Test cases for RAG ingestion pipeline"""

    @patch('rag.vector_store')
    @patch('os.listdir')
    @patch('os.path.join')
    def test_rag_ingestion_initialization(self, mock_join, mock_listdir, mock_vector_store):
        """Test RAG ingestion initialization with mocks"""
        # Mock file system
        mock_listdir.return_value = ['doc1.txt', 'doc2.md', 'ignore.pdf']
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        # Mock vector store
        mock_vector_store.reset_collection = Mock()
        mock_vector_store.add_documents = Mock(return_value=['id1', 'id2'])
        mock_vector_store.get = Mock(return_value={'ids': ['id1', 'id2']})
        
        # Mock document loading and splitting
        with patch('rag.TextLoader') as mock_loader, \
             patch('rag.RecursiveCharacterTextSplitter') as mock_splitter:
            
            # Mock document loader
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.txt"}
            mock_loader.return_value.load.return_value = [mock_doc]
            
            # Mock text splitter
            mock_splitter_instance = Mock()
            mock_splitter_instance.split_documents.return_value = [mock_doc]
            mock_splitter.return_value = mock_splitter_instance
            
            # Initialize RAG ingestion
            rag = RAGIngestion("test_docs")
            
            # Verify initialization
            assert rag.directory == "test_docs"
            assert len(rag.docs) == 2  # Only .txt and .md files
            assert len(rag.splits) == 1
            assert len(rag.document_ids) == 2

    def test_rag_ingestion_exception(self):
        """Test RAG ingestion exception"""
        exception = RAGIngestionException("Test error")
        assert str(exception) == "Test error"
        assert isinstance(exception, Exception)

    @patch('rag.vector_store')
    def test_load_documents_no_directory(self, mock_vector_store):
        """Test load_documents with no directory"""
        mock_vector_store.reset_collection = Mock()
        
        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.directory = ""
            rag.vector_store = mock_vector_store
            
            with pytest.raises(RAGIngestionException):
                rag.load_documents()

    @patch('rag.vector_store')
    def test_split_documents_no_docs(self, mock_vector_store):
        """Test split_documents with no documents loaded"""
        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.docs = []
            
            with pytest.raises(RAGIngestionException):
                rag.split_documents()

    @patch('rag.vector_store')
    def test_store_documents_no_splits(self, mock_vector_store):
        """Test store_documents with no splits"""
        mock_vector_store.reset_collection = Mock()
        
        with patch.object(RAGIngestion, '__init__', lambda x, y: None):
            rag = RAGIngestion.__new__(RAGIngestion)
            rag.splits = []
            rag.vector_store = mock_vector_store
            
            with pytest.raises(RAGIngestionException):
                rag.store_documents()


class TestRAGIngestionIntegration:
    """Integration tests for RAG ingestion"""

    def test_rag_ingestion_import(self):
        """Test that RAG ingestion can be imported"""
        from rag import RAGIngestion, RAGIngestionException
        assert RAGIngestion is not None
        assert RAGIngestionException is not None

    @patch('rag.vector_store')
    @patch('os.path.exists')
    def test_rag_ingestion_with_settings(self, mock_exists, mock_vector_store):
        """Test RAG ingestion uses settings correctly"""
        from config import settings
        
        # Mock vector store
        mock_vector_store.reset_collection = Mock()
        mock_vector_store.add_documents = Mock(return_value=[])
        mock_vector_store.get = Mock(return_value={'ids': []})
        
        # Mock file system
        mock_exists.return_value = True
        
        with patch('os.listdir', return_value=['test.txt']), \
             patch('rag.TextLoader') as mock_loader, \
             patch('rag.RecursiveCharacterTextSplitter') as mock_splitter:
            
            # Mock document loader
            mock_doc = Mock()
            mock_doc.page_content = "Test content"
            mock_doc.metadata = {"source": "test.txt"}
            mock_loader.return_value.load.return_value = [mock_doc]
            
            mock_splitter_instance = Mock()
            mock_splitter_instance.split_documents.return_value = [mock_doc]
            mock_splitter.return_value = mock_splitter_instance
            
            # Test that settings are used
            rag = RAGIngestion("test_docs")
            
            # Verify settings were used in text splitter
            mock_splitter.assert_called_once_with(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
                add_start_index=True
            )