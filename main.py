import logging
import os
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
CLASS_NAMES = {0: "stay", 1: "churn"}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "telco-churn-classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_RETRIES = int(os.getenv("MLFLOW_RETRIES", "3"))

log = logging.getLogger("uvicorn.error")
ml_model = {}


def load_from_registry():
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    info = {
        "source": "mlflow-registry",
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": int(version.version),
        "run_id": version.run_id,
        "model_type": version.tags.get("model_type"),
        "description": version.description,
    }
    return pipeline, info


def mlflow_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{MLFLOW_TRACKING_URI}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def load_model():
    for attempt in range(1, MLFLOW_RETRIES + 1):
        try:
            if not mlflow_is_up():
                raise ConnectionError(f"MLflow server {MLFLOW_TRACKING_URI} is not reachable")
            return load_from_registry()
        except Exception as exc:  # server not up yet, model not registered, ...
            log.warning("MLflow load failed (attempt %d/%d): %s", attempt, MLFLOW_RETRIES, exc)
            time.sleep(2)
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found in MLflow and no local model.joblib. Run `python train.py` first.")
    log.warning("Using local fallback %s", MODEL_PATH.name)
    return joblib.load(MODEL_PATH), {"source": "local-file", "path": MODEL_PATH.name, "model_name": MODEL_NAME}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["pipeline"], ml_model["info"] = load_model()
    log.info("Model loaded: %s", ml_model["info"])
    yield
    ml_model.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Predicts whether a telecom customer will **churn** (leave the company). "
        "Model: scikit-learn pipeline (imputer + StandardScaler + OneHotEncoder + classifier) "
        "trained on the IBM Telco Customer Churn dataset. "
        "The model is served from the MLflow Model Registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    SeniorCitizen: YesNo = Field(description="Customer is 65 or older")
    Partner: YesNo = Field(description="Customer has a partner")
    Dependents: YesNo = Field(description="Customer has dependents")
    tenure: int = Field(ge=0, le=120, description="Months the customer has stayed with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge, USD")
    TotalCharges: float = Field(ge=0, description="Total amount charged so far, USD")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "SeniorCitizen": "No", "Partner": "No", "Dependents": "No", "tenure": 2,
                "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
                "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 190.0,
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = customer stays, 1 = customer churns")
    label: Literal["stay", "churn"]
    churn_probability: float = Field(description="Predicted probability of churn")


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def model_info():
    if "info" not in ml_model:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return ml_model["info"]


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    X = pd.DataFrame([customer.model_dump()])
    pred = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0][1])
    return PredictionResponse(prediction=pred, label=CLASS_NAMES[pred], churn_probability=round(proba, 4))
