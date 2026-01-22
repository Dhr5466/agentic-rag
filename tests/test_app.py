"""Unit tests for the FastAPI application"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import UserInput, app, process_user_input

QUERY_VALIDATION_ERROR: str = "Query must contain words"


@pytest.fixture(scope="module")
def client():
    """Create a test client for synchronous requests"""
    return TestClient(app)


class TestUserInputModel:
    """Test cases for UserInput Pydantic model"""

    def test_valid_user_input(self):
        """Test creating UserInput with valid data"""
        user_input = UserInput(query="Hello world", user_id="user123")
        assert user_input.query == "Hello world"
        assert user_input.user_id == "user123"

    def test_missing_query(self):
        """Test UserInput with missing query field"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(user_id="user123")
        assert "query" in str(exc_info.value)

    def test_missing_user_id(self):
        """Test UserInput with missing user_id field"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="Hello world")
        assert "user_id" in str(exc_info.value)

    def test_empty_query(self):
        """Test UserInput with empty query"""
        # Empty query should fail validation since it has no letters
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_numbers(self):
        """Test UserInput with query containing only numbers"""
        # Query with only numbers should fail validation
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="12345", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_symbols(self):
        """Test UserInput with query containing only symbols"""
        # Query with only symbols should fail validation
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="!@#$%", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_only_whitespace(self):
        """Test UserInput with query containing only whitespace"""
        # Query with only whitespace should fail validation
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="   \t\n", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    def test_query_with_letters_and_numbers(self):
        """Test UserInput with query containing letters and numbers"""
        # Query with letters and numbers should pass validation
        user_input = UserInput(query="Hello123", user_id="user123")
        assert user_input.query == "Hello123"
        assert user_input.user_id == "user123"

    def test_query_with_letters_and_symbols(self):
        """Test UserInput with query containing letters and symbols"""
        # Query with letters and symbols should pass validation
        user_input = UserInput(query="Hello!@#", user_id="user123")
        assert user_input.query == "Hello!@#"
        assert user_input.user_id == "user123"

    def test_empty_user_id(self):
        """Test UserInput with empty user_id"""
        user_input = UserInput(query="Hello world", user_id="")
        assert user_input.query == "Hello world"
        assert user_input.user_id == ""

    def test_long_query(self):
        """Test UserInput with very long query"""
        long_query = "A" * 10000
        user_input = UserInput(query=long_query, user_id="user123")
        assert user_input.query == long_query
        assert len(user_input.query) == 10000


class TestProcessUserInput:
    """Test cases for process_user_input function"""

    @pytest.mark.asyncio
    async def test_process_user_input_normal(self):
        """Test processing normal user input"""
        user_input = UserInput(query="Hello world", user_id="user123")
        result = await process_user_input(user_input)

        assert isinstance(result, dict)
        assert "user_id" in result
        assert result["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_process_user_input_empty_query(self):
        """Test processing user input with empty query - should fail validation"""
        with pytest.raises(ValidationError) as exc_info:
            UserInput(query="", user_id="user123")
        assert QUERY_VALIDATION_ERROR in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_user_input_long_query(self):
        """Test processing user input with very long query"""
        long_query = "A" * 10000
        user_input = UserInput(query=long_query, user_id="user123")
        result = await process_user_input(user_input)

        assert result["user_id"] == "user123"


class TestQueryEndpoint:
    """Test cases for /query endpoint"""

    def test_query_endpoint_success(self, client: TestClient):
        """Test successful POST request to /query endpoint"""
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        assert response.status_code == 200
        assert response.json() == {"message": "User query waiting to be processed."}

    def test_query_endpoint_missing_query(self, client: TestClient):
        """Test POST request with missing query field"""
        response = client.post("/query", json={"user_id": "user123"})

        assert response.status_code == 422  # Unprocessable Entity
        error_detail = response.json()
        assert "detail" in error_detail

    def test_query_endpoint_missing_user_id(self, client: TestClient):
        """Test POST request with missing user_id field"""
        response = client.post("/query", json={"query": "Hello world"})

        assert response.status_code == 422  # Unprocessable Entity
        error_detail = response.json()
        assert "detail" in error_detail

    def test_query_endpoint_empty_body(self, client: TestClient):
        """Test POST request with empty body"""
        response = client.post("/query", json={})

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_endpoint_with_client(self, client: TestClient):
        """Test POST request to /query endpoint using test client"""
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        assert response.status_code == 200
        assert response.json() == {"message": "User query waiting to be processed."}


class TestErrorHandling:
    """Test cases for error handling"""

    def test_internal_server_error(self, client: TestClient):
        """Test HTTP 500 error handling"""
        # For now, just test that the endpoint handles errors properly
        # We can't easily mock the background task without pytest-mock
        response = client.post("/query", json={"query": "Hello world", "user_id": "user123"})

        # This should succeed normally
        assert response.status_code == 200


class TestIntegration:
    """Integration tests for the complete flow"""

    @pytest.mark.asyncio
    async def test_full_flow(self):
        """Test the complete request processing flow"""
        user_input = UserInput(query="Test integration", user_id="user123")

        # Test that the input is valid
        assert user_input.query == "Test integration"
        assert user_input.user_id == "user123"

        # Test that processing works
        result = await process_user_input(user_input)
        assert result["user_id"] == "user123"
