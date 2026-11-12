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

# Render's free plan: a tenth of a CPU and 512 MB of memory.
# - one inference thread uses that CPU fully; more would only queue for it
# - one photo at a time (the rest wait): 12 photo-sized uploads at once took
#   the server down, each holding its decoded image and the model's memory
# - glibc otherwise keeps a memory pool per thread and holds on to freed memory
# - images over 16 million pixels are refused; photos from the web app are
#   1600 px at most (2.6 million), and one 48 MP photo would fill the memory
# PORT is what Render tells the app to listen on.
ENV ORT_THREADS=1 \
    MAX_CONCURRENT_PREDICTIONS=1 \
    MALLOC_ARENA_MAX=2 \
    MAX_IMAGE_PIXELS=16000000 \
    PORT=10000

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 10000
# Keep-alive longer than any proxy in front (see backend/README.md, "Run").
CMD exec uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 75
