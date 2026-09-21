from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
from routers import image_router, admin_router

app = FastAPI(title="PixRevive API", description="AI-powered image restoration backend")

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
    colorizer_ready = False
    fast_models_ready = False
    try:
        from services.image_service import pipeline
        colorizer_ready = pipeline.is_colorizer_ready()
        fast_models_ready = pipeline.is_ready()
    except Exception:
        logging.getLogger(__name__).exception("Health check could not query the pipeline.")

    return {
        "status": "ok",
        "url": "http://localhost:8000",
        "models": {
            "fast_models": fast_models_ready,
            "colorizer": colorizer_ready,
            "ddcolor": colorizer_ready,
        },
        # Reflect real loader state — previously hard-coded True, so the frontend
        # could show "ready" while every request would still block on load.
        "ready": fast_models_ready,
    }
