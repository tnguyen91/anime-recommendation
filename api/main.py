import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.app_state import AppState, init_app_state
from api.auth.router import router as auth_router
from api.config import MIN_LIKES_ANIME, MIN_LIKES_USER, N_HIDDEN
from api.dependencies import get_app_state, limiter
from api.exceptions import register_exception_handlers
from api.favorites.router import router as favorites_router
from api.feedback.router import router as feedback_router
from api.inference.data_loader import load_anime_dataset
from api.inference.downloads import download_to_cache
from api.inference.model import RBM
from api.inference.preprocess import filter_data
from api.middleware import RequestIDMiddleware, configure_logging
from api.recommendations.router import router as recommendations_router
from api.settings import settings

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state = init_app_state()
    app.state.app_state = app_state

    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading anime dataset...")
    app_state.ratings_df, app_state.anime_df = load_anime_dataset()

    logger.info("Filtering data...")
    active_anime, _ = filter_data(
        app_state.ratings_df,
        min_likes_user=MIN_LIKES_USER,
        min_likes_anime=MIN_LIKES_ANIME,
    )
    app_state.anime_ids = list(active_anime) if active_anime is not None else []

    if app_state.anime_ids:
        logger.info("Initializing RBM with %d visible units...", len(app_state.anime_ids))
        app_state.rbm = RBM(n_visible=len(app_state.anime_ids), n_hidden=N_HIDDEN).to(app_state.device)

        if settings.model_uri:
            logger.info("Loading model from %s...", settings.model_uri)
            model_path = download_to_cache(settings.model_uri)
            if model_path.exists():
                ckpt = torch.load(model_path, map_location=app_state.device)
                state_dict = (
                    ckpt
                    if isinstance(ckpt, dict) and "W" in ckpt
                    else ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                )
                app_state.rbm.load_state_dict(state_dict)
                logger.info("Model loaded successfully")

    if settings.metadata_uri:
        logger.info("Loading metadata from %s...", settings.metadata_uri)
        metadata_path = download_to_cache(settings.metadata_uri)
        app_state.load_metadata(metadata_path)

    for p in settings.cache_dir.iterdir():
        if p.name.lower().endswith("anime.csv"):
            app_state.load_anime_csv(p)
            logger.info("Loaded anime data from cache: %s", p.name)
            break

    app_state.is_initialized = True
    logger.info("Startup complete")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Anime Recommendation API",
    summary="ML-powered anime recommendations using collaborative filtering.",
    description=(
        "Provides personalised anime recommendations backed by a Restricted "
        "Boltzmann Machine trained on 1.3 M+ user interactions. Includes JWT "
        "authentication, favourites management, and a feedback loop for "
        "continuous recommendation improvement."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)
app.add_middleware(RequestIDMiddleware)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
register_exception_handlers(app)

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(recommendations_router)
v1_router.include_router(auth_router)
v1_router.include_router(favorites_router)
v1_router.include_router(feedback_router)
app.include_router(v1_router)


@app.get("/health", tags=["Health"])
def health(app_state: AppState = Depends(get_app_state)):
    return app_state.get_health_status()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/health")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
