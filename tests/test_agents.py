"""agents 模块的单元测试"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document


class TestResponderAgent:
    """responder agent 的测试用例"""

    def test_responder_agent_exists(self):
        """测试 responder agent 是否正确初始化"""
        from agents.responder import responder_agent
        assert responder_agent is not None
        assert hasattr(responder_agent, 'invoke')

    def test_responder_agent_prompt(self):
        """测试 responder agent 提示词内容"""
        from agents.responder import RESPONDER_AGENT_PROMPT
        assert "零售助手" in RESPONDER_AGENT_PROMPT.lower()
        assert "generate_response" in RESPONDER_AGENT_PROMPT

    def test_generate_response_tool_exists(self):
        """测试 generate_response 工具是否存在并具有正确的属性"""
        from agents.responder import generate_response
        assert generate_response is not None
        assert hasattr(generate_response, 'name')
        assert hasattr(generate_response, 'description')


class TestRetrieverAgent:
    """retriever agent 的测试用例"""

    def test_retriever_agent_exists(self):
        """测试 retriever agent 是否正确初始化"""
        from agents.retriever import retriever_agent
        assert retriever_agent is not None
        assert hasattr(retriever_agent, 'invoke')#检查一个对象有没有某个属性 / 方法。

    def test_retriever_agent_prompt(self):
        """测试 retriever agent 提示词内容"""
        from agents.retriever import RETRIEVER_AGENT_PROMPT
        assert "上下文检索工具" in RETRIEVER_AGENT_PROMPT
        assert "最相关的文档块" in RETRIEVER_AGENT_PROMPT
        assert "未找到匹配的文档" in RETRIEVER_AGENT_PROMPT

    def test_retrieve_context_tool_exists(self):
        """测试 retrieve_context 工具是否存在并具有正确的属性"""
        from agents.retriever import retrieve_context
        assert retrieve_context is not None
        assert hasattr(retrieve_context, 'name')
        assert hasattr(retrieve_context, 'description')

    @patch('agents.retriever.vector_store')
    @patch('agents.retriever.settings')
    def test_retrieve_context_function_logic(self, mock_settings, mock_vector_store):
        """使用 mock 测试 retrieve_context 函数逻辑"""
        # 导入实际函数，而不是工具包装器
        from agents.retriever import retrieve_context

        # 设置 mock
        mock_settings.TOP_K = 3
        mock_doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt"}
        )
        mock_vector_store.similarity_search.return_value = [mock_doc]

        # 从工具中获取底层函数
        func = retrieve_context.func
        result, docs = func("test query")

        assert isinstance(result, str)
        assert "Test content" in result
        assert "test.txt" in result
        assert len(docs) == 1
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=3)

    @patch('agents.retriever.vector_store')
    def test_retrieve_context_empty_results(self, mock_vector_store):
        """测试 retrieve_context 在没有结果时的情况"""
        from agents.retriever import retrieve_context

        mock_vector_store.similarity_search.return_value = []

        # 从工具中获取底层函数
        func = retrieve_context.func
        result, docs = func("nonexistent query")

        assert result == ""
        assert len(docs) == 0


class TestAgentsIntegration:
    """agents 模块的集成测试"""

    def test_agents_module_imports(self):
        """测试 agents 模块导入是否正常工作"""
        from agents import chat_model
        assert chat_model is not None

        from agents.responder import responder_agent
        from agents.retriever import retriever_agent
        assert responder_agent is not None
        assert retriever_agent is not None

    def test_tools_are_properly_configured(self):
        """测试工具是否在 agents 中正确配置"""
        from agents.responder import responder_agent, generate_response
        from agents.retriever import retriever_agent, retrieve_context

        # 检查工具是否存在
        assert generate_response is not None
        assert retrieve_context is not None

        # 检查 agents 是否已编译为状态图（LangGraph agents）
        assert responder_agent is not None
        assert retriever_agent is not None
        assert hasattr(responder_agent, 'invoke')
        assert hasattr(retriever_agent, 'invoke')

    @patch('agents.retriever.vector_store')
    def test_vector_store_integration(self, mock_vector_store):
        """测试 retriever agent 是否与向量存储集成"""
        from agents.retriever import retrieve_context

        mock_doc = Document(
            page_content="Product X costs $100",
            metadata={"source": "prices.txt"}
        )
        mock_vector_store.similarity_search.return_value = [mock_doc]

        # 测试底层函数
        func = retrieve_context.func
        result, docs = func("What is the price of Product X?")
        
        assert "Product X costs $100" in result
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "prices.txt"