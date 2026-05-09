"""
data_loader.py
--------------
Loads and preprocesses the UCI Cleveland Heart Disease dataset.
Handles missing values, categorical encoding, and stratified train/val/test splits.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# UCI Cleveland Heart Disease — 14 standard features
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


@dataclass
class HeartData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: list[str]


def load_raw(local_path: str | None = None) -> pd.DataFrame:
    """Load the raw Cleveland dataset from disk or from UCI."""
    source = local_path or DATA_URL
    df = pd.read_csv(source, names=FEATURE_NAMES, na_values="?")
    # Binarize target: 0 = no disease, 1 = disease (any severity)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with column medians (only ca, thal typically have NaNs)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def split_and_scale(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.15, seed: int = 42
) -> HeartData:
    """Stratified split + standardize numeric columns. No leakage: scaler fit on train only."""
    y = df["target"]
    X = df.drop(columns=["target"])

    # First split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # Then carve out a validation set from the remainder
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        stratify=y_trainval,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_val[NUMERIC_FEATURES] = scaler.transform(X_val[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    return HeartData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        feature_names=list(X.columns),
    )


def get_data(local_path: str | None = None) -> HeartData:
    """
    One-shot loader: raw → clean → split → scale.

    If `local_path` is None, prefers `data/cleveland.csv` next to the repo root
    when present; otherwise falls back to the UCI URL.
    """
    if local_path is None:
        default = Path(__file__).resolve().parents[1] / "data" / "cleveland.csv"
        if default.exists():
            local_path = str(default)
    return split_and_scale(clean(load_raw(local_path)))


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "cleveland.csv"
    data = get_data(str(data_path) if data_path.exists() else None)
    print(f"Train: {data.X_train.shape} | Val: {data.X_val.shape} | Test: {data.X_test.shape}")
    print(f"Class balance (train): {data.y_train.value_counts(normalize=True).to_dict()}")
