#!/usr/bin/env python3
"""Full training pipeline: load -> preprocess -> train -> evaluate -> explain -> save."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import feature_columns, get_feature_target, load_raw
from src.evaluate import (
    evaluate_models,
    plot_confusion,
    plot_pr,
    plot_roc,
    save_metrics,
    select_best_model,
)
from src.explain import plot_shap_summary
from src.train import find_best_threshold, split_data, train_all

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    print("Loading data...")
    df = load_raw()
    X, y = get_feature_target(df)
    numeric, categorical = feature_columns(X)
    print(f"Features: {X.shape[1]} ({len(numeric)} numeric, {len(categorical)} categorical)")
    print(f"Churn rate: {y.mean():.3%} ({y.sum()} / {len(y)})")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train={len(X_train)}, Test={len(X_test)}")

    print("\nTraining models (LR, RF, XGBoost/GBM)...")
    fitted = train_all(X_train, y_train, numeric, categorical)

    # Threshold tuning on train probabilities (avoid leaking test labels into threshold)
    print("Tuning decision thresholds for F1 on training set...")
    thresholds = {}
    for name, pipe in fitted.items():
        proba_tr = pipe.predict_proba(X_train)[:, 1]
        thresholds[name] = find_best_threshold(y_train, proba_tr, metric="f1")
        print(f"  {name}: best_threshold={thresholds[name]:.2f}")

    print("\nEvaluating on held-out test set...")
    results = evaluate_models(fitted, X_test, y_test, thresholds=thresholds)

    for name, res in results.items():
        print(
            f"  {name}: ROC-AUC={res['roc_auc']:.4f}  PR-AUC={res['pr_auc']:.4f}  "
            f"F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}"
        )

    best_name = select_best_model(results, key="roc_auc")
    best = results[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # Plots
    print("Saving figures...")
    plot_roc(results, y_test)
    plot_pr(results, y_test)
    plot_confusion(
        y_test,
        best["y_proba"],
        best["threshold"],
        best_name,
    )

    # SHAP on best model using a train subsample (transformed inside)
    print("Computing explainability plot...")
    shap_path, method = plot_shap_summary(
        fitted[best_name], X_train.sample(n=min(300, len(X_train)), random_state=42), best_name
    )
    print(f"  explainability via {method}: {shap_path}")

    # Persist best pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "churn_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": fitted[best_name],
            "model_name": best_name,
            "threshold": best["threshold"],
            "numeric_features": numeric,
            "categorical_features": categorical,
            "metrics": {k: v for k, v in best.items() if k != "y_proba"},
        },
        model_path,
        compress=3,
    )
    print(f"Saved pipeline -> {model_path}")
    # ASCII sidecar for environments that cannot commit binary joblib via API
    import base64
    b64_path = model_path.with_suffix(model_path.suffix + ".b64")
    b64_path.write_text(base64.b64encode(model_path.read_bytes()).decode("ascii"))
    print(f"Saved b64 sidecar -> {b64_path}")

    # Also save comparison artifact
    comparison = {
        name: {k: v for k, v in res.items() if k != "y_proba"}
        for name, res in results.items()
    }
    meta = {
        "best_model": best_name,
        "selection_criterion": "roc_auc",
        "imbalance_handling": (
            "class_weight='balanced' (LR/RF); scale_pos_weight (XGBoost); "
            "F1-optimized decision threshold tuned on train set"
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "thresholds": thresholds,
    }
    save_metrics(results, extra=meta)
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"models": comparison, "meta": meta}, f, indent=2)

    print(f"\nMetrics written to {REPORTS_DIR / 'metrics.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
