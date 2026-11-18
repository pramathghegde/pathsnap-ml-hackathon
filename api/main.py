
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction Service",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionRequest(BaseModel):
    trip_id: str


@app.get("/", response_model=HealthResponse)
def home():
    return {
        "status": "running",
        "service": "PathSnap API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/predict")
def predict_route(request: PredictionRequest):
    # Actual model and HMM integration will be added
    # when the other team members provide their components.

    return {
        "status": "not_ready",
        "trip_id": request.trip_id,
        "message": "Prediction engine is awaiting model integration."
    }
