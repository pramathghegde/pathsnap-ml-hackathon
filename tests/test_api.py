
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post(
        "/api/v1/predict",
        json={"trip_id": "00000000"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "not_ready"
    assert response.json()["trip_id"] == "00000000"


def test_prediction_invalid_request():
    response = client.post(
        "/api/v1/predict",
        json={}
    )

    assert response.status_code == 422
