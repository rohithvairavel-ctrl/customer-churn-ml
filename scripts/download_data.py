#!/usr/bin/env python3
"""Download IBM Telco Customer Churn dataset into data/raw/."""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = RAW_DIR / "telco_customer_churn.csv"
SAMPLE_PATH = RAW_DIR / "telco_sample.csv"

# Public mirrors of the IBM Sample Telco Customer Churn dataset
URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in URLS:
        try:
            print(f"Downloading from:\n  {url}")
            urlretrieve(url, OUT_PATH)
            import pandas as pd

            df = pd.read_csv(OUT_PATH)
            required = {"Churn", "tenure", "MonthlyCharges"}
            if not required.issubset(df.columns):
                raise ValueError(f"Unexpected columns: {list(df.columns)[:10]}...")
            print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {OUT_PATH}")
            df.head(200).to_csv(SAMPLE_PATH, index=False)
            print(f"Wrote sample ({len(df.head(200))} rows) -> {SAMPLE_PATH}")
            return OUT_PATH
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
            continue
    raise SystemExit(f"All download URLs failed. Last error: {last_err}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    download()
