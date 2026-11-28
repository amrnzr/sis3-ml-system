import json
import warnings
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_customer_churn.csv"
MODEL_PATH = BASE_DIR / "model.joblib" 
METRICS_PATH = BASE_DIR / "metrics.json"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-classifier"
CHAMPION_ALIAS = "champion"
RANDOM_STATE = 42
warnings.filterwarnings("ignore", message=".*Integer columns.*")
TEST_SIZE = 0.2

DROPPED = ["customerID", "gender"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE,
    ),
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = (df["Churn"] == "Yes").astype(int)
    return df[FEATURES], y


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, list]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}, confusion_matrix(y_test, pred).tolist()


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI}, experiment: {EXPERIMENT_NAME}")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Dataset: {len(X)} rows, {X.shape[1]} features, churn rate = {y.mean():.3f}")

    results = {}
    for name, clf in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(clf).fit(X_train, y_train)
            metrics, cm = evaluate(pipe, X_test, y_test)

            mlflow.log_params({
                "model_type": name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_features": len(FEATURES),
                "n_train_rows": len(X_train),
            })
            mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
            mlflow.log_metrics(metrics)
            model_info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=infer_signature(X_train, pipe.predict(X_train)),
                input_example=X_train.head(3),
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
            with tempfile.TemporaryDirectory() as tmp:
                joblib.dump(pipe, Path(tmp) / "model.joblib")
                (Path(tmp) / "confusion_matrix.json").write_text(
                    json.dumps({"labels": ["stay", "churn"], "matrix": cm}, indent=2))
                mlflow.log_artifacts(tmp, artifact_path="files")
            mlflow.set_tags({"dataset": "IBM Telco Customer Churn", "task": "binary classification"})

            results[name] = {"run_id": run.info.run_id, "model_uri": model_info.model_uri, "metrics": metrics}
            print(f"{name:18s} run_id={run.info.run_id}  {metrics}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    version = mlflow.register_model(results[best_name]["model_uri"], REGISTERED_MODEL_NAME)

    client = MlflowClient()
    client.update_registered_model(
        REGISTERED_MODEL_NAME,
        description="Predicts whether a telecom customer will churn. IBM Telco Customer Churn dataset.",
    )
    client.update_model_version(
        REGISTERED_MODEL_NAME, version.version,
        description=f"{best_name}, ROC-AUC={results[best_name]['metrics']['roc_auc']} on 20% hold-out",
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "model_type", best_name)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    print(f"\nRegistered {best_name} as '{REGISTERED_MODEL_NAME}' version {version.version} "
          f"(alias @{CHAMPION_ALIAS})")

    joblib.dump(mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "experiment": EXPERIMENT_NAME,
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": int(version.version),
        "selected_model": best_name,
        "runs": results,
    }, indent=2))


if __name__ == "__main__":
    main()
