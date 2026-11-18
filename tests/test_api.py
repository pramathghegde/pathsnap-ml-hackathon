
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

    data = response.json()

    assert data["status"] == "success"
    assert data["trip_id"] == "00000000"
    assert len(data["predicted_route"]) > 0
    assert len(data["route_geometry"]) > 1
    assert data["metrics"]["num_observations"] > 0
    assert data["metrics"]["num_route_arcs"] > 0
    assert data["metrics"]["disconnect_count"] == 0


def test_prediction_invalid_request():
    response = client.post(
        "/api/v1/predict",
        json={}
    )

    assert response.status_code == 422


def test_prediction_unknown_trip():
    response = client.post(
        "/api/v1/predict",
        json={"trip_id": "99999999"}
    )

    assert response.status_code == 404
