# The price API (backend/) with the model inside, for Render or any container
# host. Build from the repo root:
#
#   docker build -f deploy/api.Dockerfile -t price-api .
#   docker run -p 8000:10000 price-api        # then http://localhost:8000/health
#
# The model (112 MB) is downloaded from the v0.1-model GitHub release while the
# image is built and checked against its sha256, so the repo doesn't carry it
# and the container starts without needing the network.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first and the model second: both stay cached while the code changes.
COPY backend/requirements.txt backend/
RUN pip install -r backend/requirements.txt

COPY backend/download_model.py backend/
RUN python backend/download_model.py

# Only the model code the API imports; the rest of model/ is training code (PyTorch).
COPY model/__init__.py model/predict_onnx.py model/preprocessing.py model/price_range.py model/
COPY backend/ backend/

# Render's free plan gives the container a tenth of a CPU: one inference thread
# uses it fully, more would only queue for it. PORT is what Render tells the
# app to listen on.
ENV ORT_THREADS=1 \
    PORT=10000

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 10000
# Keep-alive longer than any proxy in front (see backend/README.md, "Run").
CMD exec uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 75
