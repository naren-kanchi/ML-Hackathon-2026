"""Trip discovery and inference endpoints."""
from __future__ import annotations

import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import APIRouter, HTTPException

from backend.services.map_matching import evaluate_reference, predict_trip

router = APIRouter(prefix="/api/v1", tags=["PathSnap"])
TRIP_ID = re.compile(r"^[0-9]{8}$")
REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _model_bundle() -> dict[str, Any]:
    model_path = Path(__import__("os").environ.get(
        "PATHSNAP_MODEL_PATH", str(REPO_ROOT / "artifacts" / "models" / "histgradientboosting.joblib")
    )).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def _partition_dir() -> Path:
    import os
    data_root = Path(os.environ.get("PATHSNAP_DATA_DIR", str(REPO_ROOT / "Dataset"))).expanduser().resolve()
    partition = os.environ.get("PATHSNAP_PARTITION", "student_test")
    if partition not in {"train", "student_test", "evaluator_test"}:
        raise ValueError("PATHSNAP_PARTITION must be train, student_test, or evaluator_test")
    return data_root / partition


def _trip_dir(trip_id: str) -> Path:
    if not TRIP_ID.fullmatch(trip_id):
        raise HTTPException(status_code=422, detail="trip_id must contain exactly 8 digits")
    directory = (_partition_dir() / trip_id).resolve()
    if directory.parent != _partition_dir().resolve() or not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} is not available")
    return directory


@router.get("/trips")
def list_trips() -> dict[str, Any]:
    """List available trip IDs; never returns route labels or reference files."""
    try:
        root = _partition_dir()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not root.is_dir():
        raise HTTPException(status_code=503, detail=f"Trip partition is unavailable: {root}")
    ids = sorted(path.name for path in root.iterdir()
                 if path.is_dir() and TRIP_ID.fullmatch(path.name)
                 and (path / f"{path.name}.track").is_file()
                 and (path / f"{path.name}.nodes").is_file()
                 and (path / f"{path.name}.arcs").is_file())
    return {"partition": root.name, "count": len(ids), "trip_ids": ids}


@router.get("/trips/{trip_id}")
def predict(trip_id: str) -> dict[str, Any]:
    """Return classifier and Viterbi predictions, plus aggregate metrics only."""
    trip_dir = _trip_dir(trip_id)
    try:
        started = time.perf_counter()
        prediction = predict_trip(trip_dir, _model_bundle())
        prediction["inference_time_seconds"] = time.perf_counter() - started
        # Reference route labels are loaded only after prediction completes.
        metrics = evaluate_reference(prediction, trip_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not process trip {trip_id}: {exc}") from exc
    prediction.pop("_metadata", None)
    prediction["evaluation"] = metrics
    prediction["partition"] = _partition_dir().name
    return prediction
