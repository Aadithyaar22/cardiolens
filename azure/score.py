"""
score.py
--------
Azure ML Managed Online Endpoint scoring script.

`init()` runs once when the container starts: pulls the joblib bundle from
Azure Blob Storage and warms it into memory.

`run(...)` is invoked per request: validates input, predicts risk,
emits structured logs that Azure Application Insights ingests automatically.
"""

from __future__ import annotations

import json
import logging
import os
import time
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

# Application Insights captures stdlib logging automatically inside Azure ML
logger = logging.getLogger("cardiolens.score")
logger.setLevel(logging.INFO)


_BUNDLE = None
_MODEL_VERSION = os.getenv("MODEL_VERSION", "cardiolens-v0.1")
_BLOB_CONNECTION = os.getenv("AZURE_BLOB_CONNECTION_STRING")
_BLOB_CONTAINER = os.getenv("MODEL_BLOB_CONTAINER", "cardiolens-models")
_BLOB_NAME = os.getenv("MODEL_BLOB_NAME", "champion_model.joblib")


def _load_from_blob():
    """Pull the model bundle from Azure Blob Storage."""
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(_BLOB_CONNECTION)
    blob = service.get_blob_client(container=_BLOB_CONTAINER, blob=_BLOB_NAME)
    buf = BytesIO()
    blob.download_blob().readinto(buf)
    buf.seek(0)
    return joblib.load(buf)


def init() -> None:
    """Called once at container startup."""
    global _BUNDLE
    if _BLOB_CONNECTION:
        logger.info("Loading model from Azure Blob Storage")
        _BUNDLE = _load_from_blob()
    else:
        # Fallback: model packaged with the deployment
        from pathlib import Path

        path = Path(os.getenv("AZUREML_MODEL_DIR", ".")) / "champion_model.joblib"
        logger.info(f"Loading model from local path {path}")
        _BUNDLE = joblib.load(path)
    logger.info(f"Model loaded — version {_MODEL_VERSION}")


def run(raw_data: str) -> dict:
    """Called per request. raw_data is a JSON string."""
    start = time.perf_counter()
    try:
        payload = json.loads(raw_data)
        rows = payload.get("data") or payload.get("instances") or [payload]
        df = pd.DataFrame(rows)

        scaler = _BUNDLE["scaler"]
        model = _BUNDLE["model"]
        feature_names = _BUNDLE["feature_names"]
        from src.data_loader import NUMERIC_FEATURES

        df = df[feature_names]
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

        proba = model.predict_proba(df)[:, 1].tolist()
        pred = (np.array(proba) >= 0.5).astype(int).tolist()
        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "prediction_served",
            extra={
                "custom_dimensions": {
                    "n_records": len(df),
                    "latency_ms": round(latency_ms, 2),
                    "model_version": _MODEL_VERSION,
                    "mean_proba": float(np.mean(proba)),
                }
            },
        )

        return {
            "predictions": pred,
            "probabilities": proba,
            "model_version": _MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        logger.exception("scoring_failed")
        return {"error": str(e), "model_version": _MODEL_VERSION}
