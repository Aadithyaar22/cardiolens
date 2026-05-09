"""
analytics.py
------------
Aggregation queries that power the "Patient History" dashboard tab.

All queries return JSON-serializable lists of dicts so the dashboard can
render them with zero post-processing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mongo.client import predictions_collection


def recent_predictions(limit: int = 20) -> list[dict]:
    cursor = (
        predictions_collection()
        .find({}, {"_id": 0, "patient_id": 1, "timestamp": 1, "risk_score": 1, "risk_tier": 1})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def tier_counts() -> list[dict]:
    pipeline = [
        {"$group": {"_id": "$risk_tier", "count": {"$sum": 1}}},
        {"$project": {"_id": 0, "tier": "$_id", "count": 1}},
        {"$sort": {"count": -1}},
    ]
    return list(predictions_collection().aggregate(pipeline))


def common_high_risk_features(top_k: int = 5) -> list[dict]:
    """
    Among predictions tagged High or Critical, surface which features had the
    largest mean positive SHAP contribution.
    """
    pipeline = [
        {"$match": {"risk_tier": {"$in": ["High", "Critical"]}}},
        {"$project": {"shap": {"$objectToArray": "$shap_values"}}},
        {"$unwind": "$shap"},
        {"$match": {"shap.v": {"$gt": 0}}},
        {"$group": {"_id": "$shap.k", "mean_contribution": {"$avg": "$shap.v"}}},
        {"$project": {"_id": 0, "feature": "$_id", "mean_contribution": 1}},
        {"$sort": {"mean_contribution": -1}},
        {"$limit": top_k},
    ]
    return list(predictions_collection().aggregate(pipeline))


def predictions_in_window(days: int = 7) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {
            "_id": {
                "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
            },
            "n_predictions": {"$sum": 1},
            "mean_risk": {"$avg": "$risk_score"},
        }},
        {"$project": {"_id": 0, "date": "$_id", "n_predictions": 1, "mean_risk": 1}},
        {"$sort": {"date": 1}},
    ]
    return list(predictions_collection().aggregate(pipeline))


def search_patient(patient_id: str) -> list[dict]:
    cursor = (
        predictions_collection()
        .find({"patient_id": patient_id}, {"_id": 0})
        .sort("timestamp", -1)
    )
    return list(cursor)
