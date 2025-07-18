import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Stock Price Alerter API" in response.json()["message"]


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_predict_without_training():
    """Test prediction endpoint without training"""
    response = client.post("/predict", json={"symbol": "AAPL"})
    assert response.status_code == 400
    assert "not trained" in response.json()["detail"]


def test_train_endpoint():
    """Test model training endpoint"""
    response = client.post("/train", json={"symbol": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["status"] == "training_in_progress"
