# `mongo/` — Clinical Data Layer

This subdirectory contains the MongoDB integration for CardioLens. It is designed to be **presentable as its own mini-project** during the demo / viva.

## What this layer demonstrates independently

> **"MongoDB as a clinical data layer for real-time patient risk logging and ML model governance."**

Two collections:

1. **`predictions`** — every inference produces a structured document containing the patient's vitals, the calibrated risk score, the assigned tier, the SHAP attribution vector, and a counterfactual narrative. Indexed on `(patient_id, timestamp DESC)` for fast per-patient timeline queries and on `risk_tier` for cohort-level aggregations.
2. **`model_registry`** — every trained champion writes a registry entry with the algorithm name, CV AUC, test metrics, hyperparameters, and timestamp. This is a lightweight in-house model versioning layer that mirrors what tools like MLflow do.

## Files

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic v2 schema contract for both collections |
| `client.py` | PyMongo wrapper with idempotent index creation and `log_prediction` / `register_model` helpers |
| `analytics.py` | Aggregation pipelines that power the "Patient History" dashboard tab — recent predictions, tier distribution, common high-risk feature patterns, weekly trend |

## Talking points for viva

- "I used the document model because each patient prediction is a self-contained unit with nested fields like SHAP values that don't fit cleanly into a relational schema."
- "The compound index on `(patient_id, timestamp)` makes the per-patient timeline a single index scan."
- "The `$objectToArray` aggregation in `common_high_risk_features()` lets me pivot SHAP values stored as a sub-document into a feature-by-feature analysis without restructuring the storage."
- "The `model_registry` collection is my own MLflow-lite — I can `db.model_registry.find().sort({trained_at: -1}).limit(1)` to see what's in production right now."
