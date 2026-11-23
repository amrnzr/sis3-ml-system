# MLflow tracking server + Model Registry
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN pip install --no-cache-dir mlflow==3.18.0

WORKDIR /mlflow
EXPOSE 5000

# Runs and registered models are stored in SQLite, artifacts (model files) on disk.
# /mlflow is mounted from ./mlflow_data, so the data survives container restarts.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "*"]
