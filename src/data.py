"""Data loading utilities for Telco Customer Churn."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "telco_sample.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the Telco churn CSV from disk."""
    csv_path = path or RAW_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run: python scripts/download_data.py"
        )
    df = pd.read_csv(csv_path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip strings, coerce TotalCharges, drop ID, encode target."""
    out = df.copy()
    # Strip whitespace from object/string columns
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()

    # TotalCharges is often object with blanks for brand-new customers
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])

    if TARGET_COL in out.columns:
        s = out[TARGET_COL]
        if not pd.api.types.is_numeric_dtype(s):
            mapped = s.astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
            )
            if mapped.isna().any():
                bad = s[mapped.isna()].unique()[:5]
                raise ValueError(f"Unmapped Churn values: {bad}")
            out[TARGET_COL] = mapped.astype(int)
        else:
            out[TARGET_COL] = s.astype(int)

    return out


def get_feature_target(df: pd.DataFrame):
    """Return X, y after cleaning."""
    cleaned = clean_dataframe(df)
    if TARGET_COL not in cleaned.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing")
    y = cleaned[TARGET_COL]
    X = cleaned.drop(columns=[TARGET_COL])
    return X, y


def feature_columns(X: pd.DataFrame):
    """Identify numeric vs categorical columns."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical
