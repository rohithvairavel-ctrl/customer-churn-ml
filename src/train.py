"""Model training helpers: compare LR, RF, XGBoost/GBM."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor


def make_models(random_state: int = 42) -> dict[str, Any]:
    """Return candidate classifiers with class_weight where supported."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            # scale_pos_weight set later once we know class ratio
        )
    except Exception:
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )
    return models


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_full_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def train_all(
    X_train,
    y_train,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Fit each model pipeline; set XGBoost scale_pos_weight from class ratio."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = make_models(random_state=random_state)

    # Class imbalance ratio for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale = n_neg / max(n_pos, 1)

    fitted: dict[str, Pipeline] = {}
    for name, est in models.items():
        # Fresh preprocessor per model so pipelines are independent
        prep = build_preprocessor(numeric_features, categorical_features)
        if name == "xgboost" and hasattr(est, "set_params"):
            est.set_params(scale_pos_weight=scale)
        pipe = build_full_pipeline(prep, est)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> float:
    """Scan thresholds to maximize F1 (or precision/recall balance)."""
    from sklearn.metrics import f1_score

    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t
