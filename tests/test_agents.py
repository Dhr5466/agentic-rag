"""Unit tests for agents module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document


class TestResponderAgent:
    """Test cases for responder agent"""

    def test_responder_agent_exists(self):
        """Test that responder agent is properly initialized"""
        from agents.responder import responder_agent
        assert responder_agent is not None
        assert hasattr(responder_agent, 'invoke')

    def test_responder_agent_prompt(self):
        """Test responder agent prompt content"""
        from agents.responder import RESPONDER_AGENT_PROMPT
        assert "retail assistant" in RESPONDER_AGENT_PROMPT.lower()
        assert "generate_response" in RESPONDER_AGENT_PROMPT

    def test_generate_response_tool_exists(self):
        """Test that generate_response tool exists and has correct properties"""
        from agents.responder import generate_response
        assert generate_response is not None
        assert hasattr(generate_response, 'name')
        assert hasattr(generate_response, 'description')


class TestRetrieverAgent:
    """Test cases for retriever agent"""

    def test_retriever_agent_exists(self):
        """Test that retriever agent is properly initialized"""
        from agents.retriever import retriever_agent
        assert retriever_agent is not None
        assert hasattr(retriever_agent, 'invoke')

    def test_retriever_agent_prompt(self):
        """Test retriever agent prompt content"""
        from agents.retriever import RETRIEVER_AGENT_PROMPT
        assert "context retriever tool" in RETRIEVER_AGENT_PROMPT.lower()
        assert "relevant document chunks" in RETRIEVER_AGENT_PROMPT.lower()

    def test_retrieve_context_tool_exists(self):
        """Test that retrieve_context tool exists and has correct properties"""
        from agents.retriever import retrieve_context
        assert retrieve_context is not None
        assert hasattr(retrieve_context, 'name')
        assert hasattr(retrieve_context, 'description')

    @patch('agents.retriever.vector_store')
    @patch('agents.retriever.settings')
    def test_retrieve_context_function_logic(self, mock_settings, mock_vector_store):
        """Test retrieve_context function logic with mocks"""
        # Import the actual function, not the tool wrapper
        from agents.retriever import retrieve_context
        
        # Setup mocks
        mock_settings.TOP_K = 3
        mock_doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"}
        )
        mock_vector_store.similarity_search.return_value = [mock_doc]
        
        # Get the underlying function from the tool
        func = retrieve_context.func
        result, docs = func("test query")
        
        assert isinstance(result, str)
        assert "Test content" in result
        assert "test.txt" in result
        assert len(docs) == 1
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=3)

    @patch('agents.retriever.vector_store')
    def test_retrieve_context_empty_results(self, mock_vector_store):
        """Test retrieve_context with no results"""
        from agents.retriever import retrieve_context
        
        mock_vector_store.similarity_search.return_value = []
        
        # Get the underlying function from the tool
        func = retrieve_context.func
        result, docs = func("nonexistent query")
        
        assert result == ""
        assert len(docs) == 0


class TestAgentsIntegration:
    """Integration tests for agents module"""

    def test_agents_module_imports(self):
        """Test that agents module imports work correctly"""
        from agents import chat_model
        assert chat_model is not None
        
        from agents.responder import responder_agent
        from agents.retriever import retriever_agent
        assert responder_agent is not None
        assert retriever_agent is not None

    def test_tools_are_properly_configured(self):
        """Test that tools are properly configured in agents"""
        from agents.responder import responder_agent, generate_response
        from agents.retriever import retriever_agent, retrieve_context
        
        # Check that tools exist
        assert generate_response is not None
        assert retrieve_context is not None
        
        # Check that agents are compiled state graphs (LangGraph agents)
        assert responder_agent is not None
        assert retriever_agent is not None
        assert hasattr(responder_agent, 'invoke')
        assert hasattr(retriever_agent, 'invoke')

    @patch('agents.retriever.vector_store')
    def test_vector_store_integration(self, mock_vector_store):
        """Test that retriever agent integrates with vector store"""
        from agents.retriever import retrieve_context
        
        mock_doc = Document(
            page_content="Product X costs $100",
            metadata={"source": "prices.txt"}
        )
        mock_vector_store.similarity_search.return_value = [mock_doc]
        
        # Test the underlying function
        func = retrieve_context.func
        result, docs = func("What is the price of Product X?")
        
        assert "Product X costs $100" in result
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "prices.txt"