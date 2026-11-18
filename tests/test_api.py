
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

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction engine is currently unavailable. Model integration is pending."
    )


def test_prediction_invalid_request():
    response = client.post(
        "/api/v1/predict",
        json={}
    )

    assert response.status_code == 422
def test_prediction_invalid_trip_id():
    response = client.post(
        "/api/v1/predict",
        json={"trip_id": "abc123"}
    )

    assert response.status_code == 422