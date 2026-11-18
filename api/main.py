
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from pathlib import Path

# Configure API logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pathsnap")

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction Service",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionRequest(BaseModel):
    trip_id: str = Field(
        pattern=r"^\d{8}$",
        description="Trip ID must contain exactly 8 digits"
    )


@app.get("/", response_model=HealthResponse)
def home():
    return {
        "status": "running",
        "service": "PathSnap API"
    }


@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/predict")
def predict_route(request: PredictionRequest):
    try:
        logger.info(
            "Prediction request received for trip_id=%s",
            request.trip_id
        )

        # Actual model and HMM integration will be added later.
        logger.warning(
            "Prediction engine is not ready for trip_id=%s",
            request.trip_id
        )

        raise HTTPException(
            status_code=503,
            detail="Prediction engine is currently unavailable. Model integration is pending."
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Unexpected error while processing trip_id=%s",
            request.trip_id
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred."
        )
