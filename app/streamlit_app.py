"""Streamlit app: Telco Customer Churn Prediction."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import local_contributions

MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"

DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}


@st.cache_resource
def load_artifact():
    import base64
    import io
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    b64_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".b64")
    if b64_path.exists():
        raw = base64.b64decode(b64_path.read_text().encode("ascii"))
        return joblib.load(io.BytesIO(raw))
    return None


def build_input_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values])


def main():
    st.set_page_config(page_title="Telco Churn Predictor", page_icon=":chart_with_downwards_trend:", layout="wide")
    st.title("Telco Customer Churn Predictor")
    st.markdown(
        "Predict churn probability with a trained scikit-learn **Pipeline** "
        "(preprocessing + model). Built on the IBM Telco Customer Churn dataset."
    )

    artifact = load_artifact()
    if artifact is None:
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Run `python scripts/download_data.py` then `python scripts/train.py`."
        )
        st.stop()

    pipe = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.5))
    model_name = artifact.get("model_name", "model")
    metrics = artifact.get("metrics", {})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model", model_name.replace("_", " ").title())
    col_b.metric("Decision threshold", f"{threshold:.2f}")
    if metrics:
        col_c.metric("Test ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")

    with st.expander("Test-set metrics", expanded=False):
        st.json({k: v for k, v in metrics.items() if k != "classification_report"})

    st.subheader("Customer features")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        senior = st.selectbox("Senior Citizen", [0, 1], index=0)
        partner = st.selectbox("Partner", ["Yes", "No"], index=0)
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
        tenure = st.slider("Tenure (months)", 0, 72, DEFAULTS["tenure"])
        phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=1)

    with c2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=1)
        online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=0)
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=1)
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=1)
        stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0)
        stream_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=1)

    with c3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        )
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

    values = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    if st.button("Predict churn", type="primary"):
        row = build_input_row(values)
        num_f = artifact.get("numeric_features", [])
        cat_f = artifact.get("categorical_features", [])
        expected = list(num_f) + list(cat_f)
        if expected:
            missing = [c for c in expected if c not in row.columns]
            if missing:
                st.warning(f"Missing columns filled with defaults where possible: {missing}")
            row = row.reindex(columns=expected, fill_value=0)

        proba = float(pipe.predict_proba(row)[0, 1])
        pred = int(proba >= threshold)

        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Prediction", "CHURN" if pred else "STAY")

        st.progress(min(max(proba, 0.0), 1.0))

        st.subheader("Top feature contributions")
        try:
            contribs = local_contributions(pipe, row, top_k=10)
            if contribs:
                cdf = pd.DataFrame(contribs)
                st.dataframe(cdf, use_container_width=True)
                st.bar_chart(cdf.set_index("feature")["contribution"])
            else:
                st.info("Could not compute local contributions for this model.")
        except Exception as e:
            st.warning(f"Explainability unavailable: {e}")

    st.divider()
    figs = PROJECT_ROOT / "reports" / "figures"

    def _fig(*names):
        for n in names:
            cand = figs / n
            if cand.exists():
                return str(cand)
        return None

    roc = _fig("roc_curve.png", "roc_curve.svg")
    cm = _fig("confusion_matrix.png", "confusion_matrix.svg")
    shap = _fig("shap_summary.png", "feature_importance.png", "feature_importance.svg")
    if roc or cm or shap:
        st.subheader("Model diagnostics")
        f1, f2, f3 = st.columns(3)
        if roc:
            f1.image(roc, caption="ROC")
        if cm:
            f2.image(cm, caption="Confusion matrix")
        if shap:
            f3.image(shap, caption="SHAP / importance")


if __name__ == "__main__":
    main()
