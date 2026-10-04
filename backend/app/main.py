"""
Amazon Vision Price Predictor API (Phase 4).

Run from the repo root (so `model/` is importable):

    uvicorn backend.app.main:app --reload

    GET  /health    is the model loaded, which version
    POST /predict   multipart upload, field "file": a product photo
    GET  /docs      interactive API docs

The model loads once at startup (lifespan), not per request, and the app
refuses to start if it's missing or doesn't match its model card.
/predict is a plain `def`, so FastAPI runs it in a worker thread and the
CPU-bound inference doesn't block other requests.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import Settings
from backend.app.images import UploadError, check_declared_type, open_image, read_upload
from backend.app.predictor import Predictor
from backend.app.schemas import ErrorResponse, Health, Prediction

log = logging.getLogger("uvicorn.error")

ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty, corrupt or not an image"},
    413: {"model": ErrorResponse, "description": "File or image too large"},
    415: {"model": ErrorResponse, "description": "Not a JPEG, PNG or WebP"},
}


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        p = Predictor(settings)
        app.state.predictor = p
        log.info("Loaded %s (%s, sha256 %s...), price ranges at %.0f%% coverage",
                 settings.model_path.name, p.version, p.sha256[:12],
                 100 * p.calibration["coverage"])
        yield

    app = FastAPI(
        title="Amazon Vision Price Predictor",
        version="0.1.0",
        description="Estimates a product's price from a photo, with a calibrated price range.",
        lifespan=lifespan,
    )

    # Added before CORS on purpose: the last middleware added runs first, so
    # CORS wraps this one and even a 413 reaches the browser with CORS headers.
    @app.middleware("http")
    async def refuse_oversized_uploads(request: Request, call_next):
        # Turn away a declared-too-big upload before reading its body. A body
        # sent without Content-Length is still capped in read_upload().
        length = request.headers.get("content-length", "")
        if (request.url.path == "/predict" and length.isdigit()
                and int(length) > settings.max_upload_bytes + 64 * 1024):  # room for form headers
            return JSONResponse(status_code=413,
                                content={"detail": f"File is larger than {settings.max_upload_mb:g} MB."})
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": app.title, "docs": "/docs", "health": "/health", "predict": "POST /predict"}

    @app.get("/health", response_model=Health)
    def health(request: Request):
        p = request.app.state.predictor
        return Health(status="ok", model_version=p.version, model_sha256=p.sha256,
                      input_size=p.input_size, range_coverage=p.calibration["coverage"])

    @app.post("/predict", response_model=Prediction, responses=ERRORS)
    def predict(request: Request,
                file: UploadFile = File(description="Product photo: JPEG, PNG or WebP")):
        try:
            check_declared_type(file.content_type)
            data = read_upload(file.file, settings.max_upload_bytes)
            img = open_image(data, settings.max_image_pixels)
        except UploadError as e:
            raise HTTPException(status_code=e.status, detail=e.detail)
        with img:
            return request.app.state.predictor.predict(img)

    return app


app = create_app()
