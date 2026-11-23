from fastapi.testclient import TestClient

from main import app

EXAMPLE = app.openapi()["components"]["schemas"]["Customer"]["examples"][0]

LOYAL_CUSTOMER = {
    **EXAMPLE, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
    "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
}


def test_root():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json() == {"message": "ML API is running"}


def test_predict():
    with TestClient(app) as client:
        r = client.post("/predict", json=EXAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in (0, 1)
        assert body["label"] in ("stay", "churn")
        assert 0.0 <= body["churn_probability"] <= 1.0


def test_new_fiber_customer_riskier_than_loyal_one():
    with TestClient(app) as client:
        risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
        loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()["churn_probability"]
        assert risky > loyal


def test_predict_rejects_invalid_input():
    with TestClient(app) as client:
        r = client.post("/predict", json={**EXAMPLE, "Contract": "Ten years"})
        assert r.status_code == 422


def test_docs_available():
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200


def test_model_info():
    with TestClient(app) as client:
        r = client.get("/model-info")
        assert r.status_code == 200
        assert r.json()["model_name"] == "telco-churn-classifier"
