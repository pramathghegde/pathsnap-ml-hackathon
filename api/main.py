
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.loader import load_trip
from src.models.predict import load_predictor
from src.decoding.reconstruction import reconstruct_route


# --------------------------------------------------
# Paths and logging
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Student folder"
    / "student_test"
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("pathsnap")


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_predictor():
    logger.info("Loading PathSnap trained model")
    predictor = load_predictor()
    logger.info("Model loaded successfully")
    return predictor


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/api/v1/predict")
def predict_route(request: PredictionRequest):

    trip_id = request.trip_id
    start_time = time.perf_counter()

    try:
        logger.info(
            "Prediction request received: trip_id=%s",
            trip_id
        )

        # Only allow trips in the student-test partition.
        trip_dir = DATA_DIR / trip_id

        if not trip_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail="Trip not found in the student-test dataset."
            )

        # Load GPS observations and road network.
        trip = load_trip(
            trip_dir,
            partition="student_test"
        )

        # Never pass the reference route into inference.
        trip = replace(trip, route=None)

        # Load the trained classifier.
        predictor = get_predictor()

        # Run candidate generation, classification and HMM.
        result = reconstruct_route(
            trip,
            predictor=predictor
        )

        if result.diagnostics.get("status") != "success":
            raise HTTPException(
                status_code=422,
                detail="Route reconstruction could not produce a valid result."
            )

        # Convert predicted arc IDs into map coordinates.
        route_geometry = []

        for arc_id in result.reconstructed_route:
            if arc_id < 0 or arc_id >= len(trip.arcs):
                continue

            from_node, to_node = trip.arcs[arc_id]

            start_point = trip.nodes[int(from_node)]
            end_point = trip.nodes[int(to_node)]

            if not route_geometry:
                route_geometry.append([
                    float(start_point[0]),
                    float(start_point[1])
                ])

            route_geometry.append([
                float(end_point[0]),
                float(end_point[1])
            ])

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        logger.info(
            "Prediction successful: trip_id=%s, route_arcs=%s, runtime_ms=%s",
            trip_id,
            result.num_route_arcs,
            elapsed_ms
        )

        return {
            "status": "success",
            "trip_id": trip_id,
            "predicted_route": result.reconstructed_route,
            "route_geometry": route_geometry,
            "metrics": {
                "num_observations": result.num_observations,
                "num_route_arcs": result.num_route_arcs,
                "continuity_score": result.continuity_score,
                "gap_count": result.gap_count,
                "disconnect_count": result.disconnect_count,
                "runtime_ms": elapsed_ms,
                "viterbi_runtime_ms": result.diagnostics.get(
                    "viterbi_runtime_ms"
                )
            },
            "message": "Route reconstructed successfully."
        }

    except HTTPException:
        raise

    except FileNotFoundError:
        logger.exception("Trip data file missing: %s", trip_id)
        raise HTTPException(
            status_code=404,
            detail="Required trip data file was not found."
        )

    except Exception:
        logger.exception(
            "Prediction failed for trip_id=%s",
            trip_id
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during prediction."
        )
