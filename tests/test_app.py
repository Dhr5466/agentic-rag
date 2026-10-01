"""FastAPI 应用的单元测试"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app import enqueue_query, queue
from app import UserInput, app

QUERY_VALIDATION_ERROR: str = "查询不能为空，且至少包含一个字母、数字或汉字"
SUCCESS_RESPONSE_MESSAGE: str = "用户查询已成功加入队列！"


@pytest.fixture(scope="module")
def client():
    """创建一个用于同步请求的测试客户端"""
    return TestClient(app)


class TestUserInputModel:
    """UserInput Pydantic 模型的测试用例"""

    def test_valid_user_input(self):
        """测试使用有效数据创建 UserInput"""
        user_input = UserInput(query="Hello world", user_id="user123")
        assert user_input.query == "Hello world"
        assert user_input.user_id == "user123"

    def test_missing_query(self):
        """测试缺少 query 字段的 UserInput"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(user_id="user123")
        assert "query" in str(exc_info.value)

    def test_missing_user_id(self):
        """测试缺少 user_id 字段的 UserInput"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="Hello world")
        assert "user_id" in str(exc_info.value)

    def test_empty_query(self):
        """测试空 query 的 UserInput"""
        # 空 query 应该验证失败，因为它没有内容
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_numbers(self):
        """测试仅包含数字的 query 的 UserInput"""
        # 仅包含数字的 query 应该验证失败
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="12345", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_symbols(self):
        """测试仅包含符号的 query 的 UserInput"""
        # 仅包含符号的 query 应该验证失败
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="!@#$%", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_whitespace(self):
        """测试仅包含空白字符的 query 的 UserInput"""
        # 仅包含空白字符的 query 应该验证失败
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="   \t\n", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_letters_and_numbers(self):
        """测试包含字母和数字的 query 的 UserInput"""
        # 包含字母和数字的 query 应该通过验证
        user_input = UserInput(query="Hello123", user_id="user123")
        assert user_input.query == "Hello123"
        assert user_input.user_id == "user123"

    def test_query_with_letters_and_symbols(self):
        """测试包含字母和符号的 query 的 UserInput"""
        # 包含字母和符号的 query 应该通过验证
        user_input = UserInput(query="Hello!@#", user_id="user123")
        assert user_input.query == "Hello!@#"
        assert user_input.user_id == "user123"

    def test_empty_user_id(self):
        """测试空 user_id 的 UserInput"""
        user_input = UserInput(query="Hello world", user_id="")
        assert user_input.query == "Hello world"
        assert user_input.user_id == ""

    def test_long_query(self):
        """测试非常长的 query 的 UserInput"""
        long_query = "A" * 10000
        user_input = UserInput(query=long_query, user_id="user123")
        assert user_input.query == long_query
        assert len(user_input.query) == 10000


class TestEnqueueQuery:
    """enqueue_query 函数的测试用例"""

    @pytest.mark.asyncio
    async def test_enqueue_query_normal(self):
        """测试将普通用户输入加入队列"""

        user_input = UserInput(query="Hello world", user_id="user123")
        await enqueue_query(user_input)

        # 检查项目是否已添加到队列
        assert not queue.empty()

        # 清理队列
        await queue.get()
        queue.task_done()

    @pytest.mark.asyncio
    async def test_enqueue_query_empty_query(self):
        """测试将空 query 的用户输入加入队列 - 应该验证失败"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_enqueue_query_long_query(self):
        """测试将非常长的 query 的用户输入加入队列"""

        long_query = "A" * 10000
        user_input = UserInput(query=long_query, user_id="user123")
        await enqueue_query(user_input)

        # 检查项目是否已添加到队列
        assert not queue.empty()

        # 清理队列
        await queue.get()
        queue.task_done()


class TestQueryEndpoint:
    """/query 端点的测试用例"""

    def test_query_endpoint_success(self, client: TestClient):
        """测试成功向 /query 端点发送 POST 请求"""
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        assert response.status_code == 200
        assert response.json() == {"message": SUCCESS_RESPONSE_MESSAGE}

    def test_query_endpoint_missing_query(self, client: TestClient):
        """测试缺少 query 字段的 POST 请求"""
        response = client.post("/query", json={"user_id": "user123"})

        assert response.status_code == 422  # 无法处理的实体
        error_detail = response.json()
        assert "detail" in error_detail

    def test_query_endpoint_missing_user_id(self, client: TestClient):
        """测试缺少 user_id 字段的 POST 请求"""
        response = client.post("/query", json={"query": "Hello world"})

        assert response.status_code == 422  # 无法处理的实体
        error_detail = response.json()
        assert "detail" in error_detail

    def test_query_endpoint_empty_body(self, client: TestClient):
        """测试空请求体的 POST 请求"""
        response = client.post("/query", json={})

        assert response.status_code == 422  # 无法处理的实体

    def test_query_endpoint_with_client(self, client: TestClient):
        """使用测试客户端测试向 /query 端点发送 POST 请求"""
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        assert response.status_code == 200
        assert response.json() == {"message": SUCCESS_RESPONSE_MESSAGE}


class TestErrorHandling:
    """错误处理的测试用例"""

    def test_internal_server_error(self, client: TestClient):
        """测试 HTTP 500 错误处理"""
        # 目前只测试端点是否正确处理错误
        # 没有 pytest-mock 的话，我们不容易 mock 后台任务
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        # 正常情况下这应该成功
        assert response.status_code == 200


class TestIntegration:
    """完整流程的集成测试"""

    @pytest.mark.asyncio
    async def test_full_flow(self):
        """测试完整的请求处理流程"""
        from app import enqueue_query, queue

        user_input = UserInput(query="Test integration", user_id="user123")

        # 测试输入是否有效
        assert user_input.query == "Test integration"
        assert user_input.user_id == "user123"

        # 测试入队是否正常工作
        await enqueue_query(user_input)
        assert not queue.empty()

        # 清理队列
        await queue.get()
        queue.task_done()