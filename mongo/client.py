"""
client.py
---------
Thin wrapper around PyMongo. Reads MONGO_URI from environment.
Creates indexes the first time it is called (idempotent).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from pymongo import ASCENDING, DESCENDING, MongoClient

from mongo.schemas import PredictionDoc


DB_NAME = os.getenv("CARDIOLENS_DB", "cardiolens")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError(
            "MONGO_URI is not set. Add it to your .env to enable persistence."
        )
    client = MongoClient(uri, serverSelectionTimeoutMS=4000)
    client.admin.command("ping")
    _ensure_indexes(client)
    return client


def _ensure_indexes(client: MongoClient) -> None:
    db = client[DB_NAME]
    db.predictions.create_index([("patient_id", ASCENDING), ("timestamp", DESCENDING)])
    db.predictions.create_index([("risk_tier", ASCENDING)])
    db.model_registry.create_index([("version", ASCENDING)], unique=True)


def predictions_collection():
    return get_client()[DB_NAME].predictions


def registry_collection():
    return get_client()[DB_NAME].model_registry


def log_prediction(
    patient_id: str,
    raw_features: dict[str, float],
    risk_proba: float,
    risk_tier: str,
    shap_values: dict[str, float],
    counterfactual: str | None = None,
    interval: dict[str, float] | None = None,
) -> Any:
    doc = PredictionDoc(
        patient_id=patient_id,
        input_vitals=raw_features,
        risk_score=risk_proba,
        risk_tier=risk_tier,
        shap_values=shap_values,
        counterfactual=counterfactual,
        interval=interval,
    )
    return predictions_collection().insert_one(doc.model_dump()).inserted_id


def register_model(payload: dict[str, Any]) -> Any:
    return registry_collection().insert_one(payload).inserted_id
