"""Streamlit frontend for the Telco Customer Churn API.

The page collects customer data, sends it to the FastAPI service (POST /predict)
and shows the predicted churn probability.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# ---------- Sidebar: service status and model info ----------
with st.sidebar:
    st.header("Service status")
    root = api_get("/")
    if root:
        st.success(f"API: {root['message']}")
    else:
        st.error(f"API is not reachable at {API_URL}")

    info = api_get("/model-info")
    if info:
        st.subheader("Served model")
        if info.get("source") == "mlflow-registry":
            st.markdown(
                f"**{info['model_name']}** v{info['version']} (@{info['alias']})  \n"
                f"Type: {info.get('model_type')}  \n"
                f"Run: `{info['run_id'][:8]}`"
            )
            if info.get("description"):
                st.caption(info["description"])
        else:
            st.markdown(f"Local file `{info.get('path')}` (MLflow not reachable)")
    st.markdown(f"[Open MLflow UI]({MLFLOW_UI_URL})  ·  [API docs]({API_URL}/docs)")

# ---------- Main form ----------
st.title("📉 Customer Churn Predictor")
st.write("Enter the customer's data to estimate the probability that they leave the telecom company.")

with st.form("customer"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Customer")
        senior = st.selectbox("Senior citizen (65+)", ["No", "Yes"])
        partner = st.selectbox("Has a partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months with the company)", 0, 72, 2)

    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon_options = ["No", "Yes", "No internet service"]
        security = st.selectbox("Online security", addon_options)
        backup = st.selectbox("Online backup", addon_options)
        protection = st.selectbox("Device protection", addon_options)
        support = st.selectbox("Tech support", addon_options)
        tv = st.selectbox("Streaming TV", addon_options, index=1)
        movies = st.selectbox("Streaming movies", addon_options, index=1)

    with c3:
        st.subheader("Contract and billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        monthly = st.number_input("Monthly charges, USD", min_value=0.0, max_value=200.0, value=95.5, step=0.5)
        total = st.number_input("Total charges, USD", min_value=0.0, value=190.0, step=10.0,
                                help="Usually about tenure × monthly charges")

    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    payload = {
        "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
    else:
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            res = r.json()
            p = res["churn_probability"]
            m1, m2 = st.columns(2)
            m1.metric("Prediction", "Will churn" if res["label"] == "churn" else "Will stay")
            m2.metric("Churn probability", f"{p:.1%}")
            st.progress(p)
            if res["label"] == "churn":
                st.warning("High churn risk: consider a retention offer (e.g. a longer contract or a discount).")
            else:
                st.success("Low churn risk.")
            with st.expander("Request sent to the API"):
                st.json(payload)
            with st.expander("API response"):
                st.json(res)
