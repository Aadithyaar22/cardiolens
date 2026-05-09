"""
train.py
--------
End-to-end training entrypoint:
  1. Load + clean Cleveland data
  2. Cross-validate four candidate models
  3. Pick champion by mean CV AUC
  4. Refit on train+val with isotonic calibration
  5. Evaluate on test set
  6. Persist model artifact + a metrics JSON

Run from repo root:
    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import get_data
from src.evaluation import evaluate, fairness_audit, roc_curve_data
from src.models import cross_validate_all, fit_calibrated_champion


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "reports"
MODEL_PATH = ARTIFACT_DIR / "champion_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(">> Loading data")
    data = get_data()
    print(
        f"   train={data.X_train.shape}  val={data.X_val.shape}  test={data.X_test.shape}"
    )

    print(">> Cross-validating candidates")
    cv_results = cross_validate_all(data.X_train, data.y_train)
    for r in cv_results:
        print(f"   {r.name:>6}  AUC = {r.cv_auc_mean:.4f} ± {r.cv_auc_std:.4f}")

    champion = cv_results[0]
    print(f">> Champion: {champion.name}")

    X_trainval = pd.concat([data.X_train, data.X_val])
    y_trainval = pd.concat([data.y_train, data.y_val])
    model = fit_calibrated_champion(champion.name, X_trainval, y_trainval)

    print(">> Test-set evaluation")
    test_metrics = evaluate(model, data.X_test, data.y_test)
    for k, v in test_metrics.as_dict().items():
        print(f"   {k}: {v}")

    fairness = fairness_audit(model, data.X_test, data.y_test)
    print(">> Fairness audit (test set):")
    print(fairness.to_string(index=False))

    roc = roc_curve_data(model, data.X_test, data.y_test)

    artifact = {
        "champion": champion.name,
        "cv_results": [
            {"name": r.name, "auc_mean": r.cv_auc_mean, "auc_std": r.cv_auc_std}
            for r in cv_results
        ],
        "test_metrics": test_metrics.as_dict(),
        "fairness": fairness.to_dict(orient="records"),
        "roc": roc,
    }
    METRICS_PATH.write_text(json.dumps(artifact, indent=2))

    joblib.dump(
        {
            "model": model,
            "scaler": data.scaler,
            "feature_names": data.feature_names,
            "champion_name": champion.name,
        },
        MODEL_PATH,
    )
    print(f">> Saved model → {MODEL_PATH}")
    print(f">> Saved metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
