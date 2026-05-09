"""
models.py
---------
Defines the four candidate models and a champion-selection routine
based on cross-validated AUC-ROC on the training set.

Champion is then refit on train+val and calibrated using isotonic regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@dataclass
class ModelResult:
    name: str
    model: Any
    cv_auc_mean: float
    cv_auc_std: float


def candidates(seed: int = 42) -> dict[str, Any]:
    """Return a dict of named candidate classifiers tuned for tabular medical data."""
    return {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        ),
    }


def cross_validate_all(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42
) -> list[ModelResult]:
    """Run stratified k-fold CV for every candidate and return AUC summaries."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: list[ModelResult] = []
    for name, model in candidates(seed).items():
        scores = cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        results.append(
            ModelResult(
                name=name,
                model=model,
                cv_auc_mean=float(np.mean(scores)),
                cv_auc_std=float(np.std(scores)),
            )
        )
    results.sort(key=lambda r: r.cv_auc_mean, reverse=True)
    return results


def fit_calibrated_champion(
    champion_name: str,
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    seed: int = 42,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Refit champion on train+val with isotonic calibration so probabilities are honest."""
    base = candidates(seed)[champion_name]
    calibrated = CalibratedClassifierCV(base, method=method, cv=5)
    calibrated.fit(X_trainval, y_trainval)
    return calibrated
