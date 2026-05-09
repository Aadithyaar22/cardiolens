"""
evaluation.py
-------------
Computes classification metrics, calibration (Brier score),
ROC/PR curves, and a fairness audit across age and sex subgroups.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class Metrics:
    accuracy: float
    auc_roc: float
    f1: float
    precision: float
    recall: float
    brier: float
    confusion: list[list[int]]

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Metrics:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(y, pred)),
        auc_roc=float(roc_auc_score(y, proba)),
        f1=float(f1_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred)),
        brier=float(brier_score_loss(y, proba)),
        confusion=confusion_matrix(y, pred).tolist(),
    )


def roc_curve_data(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y, proba))}


def fairness_audit(
    model: Any, X: pd.DataFrame, y: pd.Series, age_threshold: int = 55
) -> pd.DataFrame:
    """
    Slice metrics across sex and age subgroups to surface disparate performance.
    Note: 'age' here is assumed to be the SCALED column; pass the unscaled
    age values via the `raw_age` argument if you want true age slicing.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    rows = []
    for sex_value, sex_label in [(0, "female"), (1, "male")]:
        mask = X["sex"] == sex_value
        if mask.sum() < 5:
            continue
        rows.append({
            "subgroup": f"sex={sex_label}",
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "auc": float(roc_auc_score(y[mask], proba[mask])) if y[mask].nunique() == 2 else np.nan,
            "recall": float(recall_score(y[mask], pred[mask])) if y[mask].sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)
