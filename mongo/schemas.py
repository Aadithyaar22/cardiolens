"""
schemas.py
----------
Pydantic models for the two MongoDB collections we maintain:

  - predictions:    one document per inference call
  - model_registry: one document per trained model version

Treat these as the canonical schema contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PredictionDoc(BaseModel):
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_vitals: dict[str, float]
    risk_score: float                                 # calibrated probability
    risk_tier: str                                    # Low | Moderate | High | Critical
    shap_values: dict[str, float]
    counterfactual: str | None = None
    interval: dict[str, float] | None = None
    model_version: str = "cardiolens-v0.1"


class ModelRegistryDoc(BaseModel):
    version: str
    champion_algorithm: str                           # logreg | rf | xgb | lgbm
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cv_auc_mean: float
    cv_auc_std: float
    test_metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    notes: str | None = None
