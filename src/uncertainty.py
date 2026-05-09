"""
uncertainty.py
--------------
Split-conformal prediction for binary classification.

Given a held-out calibration set, we compute nonconformity scores
(`1 - p_true_class`) and use the (1 - alpha) quantile to convert any
new prediction's softmax into a *prediction set* with a coverage
guarantee, plus a tight prediction interval on the risk probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalIntervals:
    point: float        # raw model probability
    lower: float        # lower bound on P(disease)
    upper: float        # upper bound on P(disease)
    coverage: float     # 1 - alpha


class ConformalBinary:
    """Split-conformal wrapper for any sklearn-style binary classifier with predict_proba."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: float | None = None

    def calibrate(self, model, X_cal: pd.DataFrame, y_cal: pd.Series) -> "ConformalBinary":
        proba = model.predict_proba(X_cal)
        # Nonconformity = 1 - probability assigned to the true class
        true_class_proba = proba[np.arange(len(y_cal)), y_cal.values]
        scores = 1.0 - true_class_proba
        n = len(scores)
        # Finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_ = float(np.quantile(scores, min(q_level, 1.0)))
        return self

    def interval(self, model, x: pd.DataFrame) -> ConformalIntervals:
        if self.q_ is None:
            raise RuntimeError("Call .calibrate(...) before .interval(...)")
        p = float(model.predict_proba(x)[0, 1])
        lower = max(0.0, p - self.q_)
        upper = min(1.0, p + self.q_)
        return ConformalIntervals(point=p, lower=lower, upper=upper, coverage=1 - self.alpha)
