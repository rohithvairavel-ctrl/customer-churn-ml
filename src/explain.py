"""SHAP / feature importance explanations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def get_feature_names(pipeline) -> list[str]:
    """Recover transformed feature names from a fitted sklearn Pipeline."""
    pre = pipeline.named_steps["preprocessor"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(pre.transform(np.zeros((1, 1))).shape[1])]


def transform_X(pipeline, X):
    return pipeline.named_steps["preprocessor"].transform(X)


def plot_shap_summary(pipeline, X_sample, model_name: str, out_path: Path | None = None, max_samples: int = 200):
    """SHAP summary for tree models; fallback to impurity / coef importance."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or FIGURES_DIR / "shap_summary.png"

    model = pipeline.named_steps["model"]
    X_t = transform_X(pipeline, X_sample)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feature_names = get_feature_names(pipeline)

    # Subsample for speed
    n = min(max_samples, X_t.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(X_t.shape[0], size=n, replace=False)
    X_s = X_t[idx]

    tree_types = ("RandomForest", "XGB", "GradientBoosting", "HistGradientBoosting")
    is_tree = any(t in type(model).__name__ for t in tree_types)

    try:
        import shap

        if is_tree:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_s)
            # Binary RF may return list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_values,
                X_s,
                feature_names=feature_names,
                show=False,
                max_display=20,
            )
            plt.title(f"SHAP Summary — {model_name.replace('_', ' ').title()}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            return out_path, "shap"
        else:
            # KernelExplainer is slow; use LinearExplainer for LR if possible
            if "Logistic" in type(model).__name__:
                explainer = shap.LinearExplainer(model, X_s)
                shap_values = explainer.shap_values(X_s)
                plt.figure(figsize=(10, 7))
                shap.summary_plot(
                    shap_values,
                    X_s,
                    feature_names=feature_names,
                    show=False,
                    max_display=20,
                )
                plt.title(f"SHAP Summary — {model_name.replace('_', ' ').title()}")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close()
                return out_path, "shap"
    except Exception as e:
        print(f"SHAP failed ({e}); falling back to feature importance.")

    # Fallback: feature importance / coefficients
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
    else:
        return None, "none"

    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        imp[order][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, "importance"


def local_contributions(pipeline, X_row, top_k: int = 10):
    """
    Return top feature contributions for a single row.
    Uses SHAP when possible, else model-specific importance * value.
    """
    model = pipeline.named_steps["model"]
    X_t = transform_X(pipeline, X_row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feature_names = get_feature_names(pipeline)

    try:
        import shap

        tree_types = ("RandomForest", "XGB", "GradientBoosting", "HistGradientBoosting")
        is_tree = any(t in type(model).__name__ for t in tree_types)
        if is_tree:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_t)
            if isinstance(sv, list):
                sv = sv[1]
            contribs = sv[0]
        elif "Logistic" in type(model).__name__:
            explainer = shap.LinearExplainer(model, X_t)
            contribs = explainer.shap_values(X_t)[0]
        else:
            raise RuntimeError("no explainer")
    except Exception:
        if hasattr(model, "feature_importances_"):
            contribs = model.feature_importances_ * X_t[0]
        elif hasattr(model, "coef_"):
            contribs = model.coef_.ravel() * X_t[0]
        else:
            return []

    order = np.argsort(np.abs(contribs))[::-1][:top_k]
    return [
        {"feature": feature_names[i], "contribution": float(contribs[i])}
        for i in order
    ]
