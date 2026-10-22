from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
from routers import image_router, admin_router
from contextlib import asynccontextmanager
from database import connect_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(title="PixRevive API", description="AI-powered image restoration backend", lifespan=lifespan)

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
os.makedirs(os.path.join(TRAINING_DATA_DIR, "bw"), exist_ok=True)
os.makedirs(os.path.join(TRAINING_DATA_DIR, "color"), exist_ok=True)


# `allow_origins=["*"]` together with `allow_credentials=True` is rejected by
# browsers — the spec forbids the `*` wildcard on credentialed requests, so
# Access-Control-Allow-Origin comes back as `*` and the fetch is blocked. Use an
# explicit allowlist instead; override with PIXREVIVE_CORS_ORIGINS (comma-separated).
_default_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:4173",
]
_env_origins = os.getenv("PIXREVIVE_CORS_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/training_data", StaticFiles(directory=TRAINING_DATA_DIR), name="training_data")

app.include_router(image_router.router, prefix="/api", tags=["image"])
app.include_router(admin_router.router, prefix="/api/admin", tags=["admin"])

@app.get("/")
async def root():
    return {"message": "Welcome to PixRevive API"}

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Per-model load truth, delegated to the pipeline.

    `health_status()` reports a boolean per model plus each failed loader's
    error, so degraded modes are visible instead of a blanket "ok".
    """
    try:
        from services.image_service import pipeline
        payload = pipeline.health_status()
    except Exception:
        logging.getLogger(__name__).exception("Health check could not query the pipeline.")
        payload = {"status": "unknown", "models": {}, "load_errors": {}, "device": None}

    return {
        # `status` is HTTP liveness ("the API answered"); per-model truth lives
        # in `models`, `ready`, and `load_errors` — degraded modes are visible
        # there without changing the original "ok" contract.
        "status": "ok",
        "url": "http://localhost:8000",
        "models": payload.get("models", {}),
        "load_errors": payload.get("load_errors", {}),
        "device": payload.get("device"),
        "ready": payload.get("status") == "ready",
    }
