import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app


class TestRequestLoggingMiddleware:
    """Test request logging middleware functionality"""

    def setup_method(self):
        self.client = TestClient(app)

    @patch("src.api.middleware.logger")
    def test_middleware_logs_successful_request(self, mock_logger):
        """Test that successful requests are logged with timing"""
        response = self.client.get("/")

        assert response.status_code == 200

        # Verify request and response logging calls
        assert mock_logger.info.call_count >= 2

        # Check request log format
        request_log = mock_logger.info.call_args_list[0][0][0]
        assert "GET /" in request_log
        assert "Client:" in request_log

        # Check response log format
        response_log = mock_logger.info.call_args_list[1][0][0]
        assert "200 -" in response_log
        assert "s" in response_log  # timing suffix

    @patch("src.api.middleware.logger")
    def test_middleware_logs_error_requests(self, mock_logger):
        """Test that error requests get enhanced logging"""
        response = self.client.get("/nonexistent")

        assert response.status_code == 404

        # Should have info logs for request/response and warning for error
        assert mock_logger.info.call_count >= 2
        assert mock_logger.warning.call_count >= 1

        # Check error logging
        error_log = mock_logger.warning.call_args_list[0][0][0]
        assert "Error response:" in error_log

    def test_middleware_preserves_functionality(self):
        """Test that middleware doesn't break existing API functionality"""
        response = self.client.get("/health")

        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"
