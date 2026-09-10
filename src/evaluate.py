"""Evaluation metrics and figure generation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def evaluate_models(
    fitted: dict,
    X_test,
    y_test,
    thresholds: dict | None = None,
) -> dict:
    """Evaluate each fitted pipeline; optional per-model thresholds."""
    thresholds = thresholds or {}
    results = {}
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        thr = thresholds.get(name, 0.5)
        results[name] = compute_metrics(y_test, proba, threshold=thr)
        results[name]["y_proba"] = proba.tolist()  # for plots; strip before JSON save
    return results


def select_best_model(results: dict, key: str = "roc_auc") -> str:
    return max(results.keys(), key=lambda n: results[n][key])


def plot_roc(results: dict, y_test, out_path: Path | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        RocCurveDisplay.from_predictions(
            y_test, res["y_proba"], name=name.replace("_", " ").title(), ax=ax
        )
    ax.set_title("ROC Curves — Telco Churn Models")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr(results: dict, y_test, out_path: Path | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / "pr_curve.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        PrecisionRecallDisplay.from_predictions(
            y_test, res["y_proba"], name=name.replace("_", " ").title(), ax=ax
        )
    ax.set_title("Precision–Recall Curves — Telco Churn Models")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion(y_true, y_proba, threshold: float, model_name: str, out_path: Path | None = None):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / "confusion_matrix.png"
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()} (t={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def metrics_for_json(results: dict) -> dict:
    """Strip non-serializable / large fields before writing JSON."""
    clean = {}
    for name, res in results.items():
        clean[name] = {
            k: v for k, v in res.items() if k != "y_proba"
        }
    return clean


def save_metrics(results: dict, path: Path | None = None, extra: dict | None = None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or REPORTS_DIR / "metrics.json"
    payload = metrics_for_json(results)
    if extra:
        payload["_meta"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
