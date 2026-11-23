# Training job: trains the models and logs them to the MLflow server
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow==3.18.0

# The project folder is mounted to /app (see docker-compose.yml),
# so train.py, data/ and the produced model.joblib / metrics.json are shared with the host.
CMD ["python", "train.py"]
