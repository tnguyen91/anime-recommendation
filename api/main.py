"""
Anime Recommendation API - Main Application.

This is the entry point for the FastAPI application. It:
1. Loads the RBM model and anime datasets at startup
2. Configures middleware (CORS, rate limiting, logging)
3. Mounts API routers for recommendations, auth, and favorites

Run locally with: uvicorn api.main:app --reload
"""
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.app_state import AppState, init_app_state
from api.config import (
    DEFAULT_TOP_N,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_ERROR,
    MIN_LIKES_ANIME,
    MIN_LIKES_USER,
    N_HIDDEN,
)
from api.inference.data_loader import load_anime_dataset
from api.inference.downloads import download_to_cache
from api.inference.model import RBM
from api.inference.preprocess import filter_data
from api.inference.recommender import get_recommendations
from api.auth.router import router as auth_router
from api.favorites.router import router as favorites_router
from api.anime_cache import set_anime_cache
from settings import settings

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

v1_router = APIRouter(prefix="/api/v1", tags=["v1"])


# =============================================================================
# Request/Response Schemas
# =============================================================================

class RecommendRequest(BaseModel):
    """Request body for getting recommendations."""
    liked_anime: list[str]
    top_n: int = DEFAULT_TOP_N
    exclude_ids: list[int] = []


class AnimeResult(BaseModel):
    """Anime details returned in search and recommendation results."""
    anime_id: int
    name: str
    title_english: str | None
    title_japanese: str | None
    image_url: str | None
    genre: list[str]
    synopsis: str | None


class RecommendResponse(BaseModel):
    """Response containing list of recommended anime."""
    recommendations: list[AnimeResult]


class SearchResponse(BaseModel):
    """Paginated search results."""
    results: list[AnimeResult]
    total: int
    limit: int
    offset: int


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_str(value) -> str:
    """Return empty string for null/NaN values, otherwise str(value)."""
    return "" if pd.isnull(value) else str(value)


def _load_model(model_path, expected_n_visible: int, device: torch.device):
    """
    Load RBM model weights from a checkpoint file.

    Handles different checkpoint formats (raw state_dict or wrapped).

    Args:
        model_path: Path to the .pth checkpoint file
        expected_n_visible: Expected number of visible units (for validation)
        device: PyTorch device to load the model onto

    Returns:
        State dict that can be loaded into the RBM model

    Raises:
        RuntimeError: If checkpoint dimensions don't match expected
    """
    ckpt = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))

    # Verify checkpoint matches expected dimensions
    for key, tensor in state_dict.items():
        if key.endswith('v_bias') and tensor.shape[0] != expected_n_visible:
            raise RuntimeError(f"Checkpoint v_bias shape {tensor.shape[0]} != expected {expected_n_visible}")
        if key.endswith('W') and tensor.shape[1] != expected_n_visible:
            raise RuntimeError(f"Checkpoint W shape {tensor.shape[1]} != expected {expected_n_visible}")

    return state_dict


def _build_anime_result(row: pd.Series, app_state: AppState) -> AnimeResult:
    """
    Build an AnimeResult from a DataFrame row and metadata.

    This is a helper to avoid code duplication between recommend and search endpoints.

    Args:
        row: DataFrame row with anime data
        app_state: Application state containing metadata

    Returns:
        AnimeResult instance
    """
    anime_id = int(row.get("anime_id", 0))
    info = app_state.get_metadata(anime_id)

    return AnimeResult(
        anime_id=anime_id,
        name=_safe_str(row.get("name")),
        title_english=_safe_str(row.get("title_english")),
        title_japanese=_safe_str(row.get("title_japanese")),
        image_url=info.get("image_url"),
        genre=info.get("genres", []),
        synopsis=info.get("synopsis")
    )


# =============================================================================
# Dependencies
# =============================================================================

def get_app_state(request: Request) -> AppState:
    """
    FastAPI dependency to get the application state.

    Usage:
        @router.get("/endpoint")
        async def endpoint(state: AppState = Depends(get_app_state)):
            ...
    """
    return request.app.state.app_state


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Initializes AppState and loads all required data:
    - Anime dataset (ratings and anime info)
    - RBM model weights
    - Anime metadata (images, synopses, etc.)
    """
    # Initialize application state
    app_state = init_app_state()
    app.state.app_state = app_state

    # Ensure cache directory exists
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load anime dataset
    logger.info("Loading anime dataset...")
    app_state.ratings_df, app_state.anime_df = load_anime_dataset()

    # Filter data for active users and anime
    logger.info("Filtering data...")
    active_anime, _ = filter_data(
        app_state.ratings_df,
        min_likes_user=MIN_LIKES_USER,
        min_likes_anime=MIN_LIKES_ANIME
    )
    app_state.anime_ids = list(active_anime) if active_anime is not None else []

    # Initialize RBM model
    if app_state.anime_ids:
        logger.info(f"Initializing RBM with {len(app_state.anime_ids)} visible units...")
        app_state.rbm = RBM(n_visible=len(app_state.anime_ids), n_hidden=N_HIDDEN).to(app_state.device)

        # Load pretrained weights if available
        if settings.model_uri:
            logger.info(f"Loading model from {settings.model_uri}...")
            model_path = download_to_cache(settings.model_uri)
            if model_path.exists():
                state_dict = _load_model(model_path, len(app_state.anime_ids), app_state.device)
                app_state.rbm.load_state_dict(state_dict)
                logger.info("Model loaded successfully")

    # Load anime metadata
    if settings.metadata_uri:
        logger.info(f"Loading metadata from {settings.metadata_uri}...")
        metadata_path = download_to_cache(settings.metadata_uri)
        app_state.load_metadata(metadata_path)

    # Load cached CSV data if available (for enriched anime info)
    for p in settings.cache_dir.iterdir():
        if p.name.lower().endswith("anime.csv"):
            app_state.load_anime_csv(p)
            logger.info(f"Loaded anime data from cache: {p.name}")
            break

    # Share metadata with favorites module (for backward compatibility)
    set_anime_cache(app_state.anime_metadata, app_state.anime_df)

    app_state.is_initialized = True
    logger.info("Startup complete")

    yield

    logger.info("Shutting down")


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="Anime Recommendation API",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their response times."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    if request.url.path not in ["/health", "/"]:
        logger.info(
            f"{request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.3f}s"
        )

    return response


# =============================================================================
# Root & Health Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Redirect to health check."""
    return await health()


@app.get("/health")
async def health(app_state: AppState = Depends(get_app_state)):
    """
    Health check endpoint for monitoring.

    Returns status of model, dataset, and metadata services.
    """
    return app_state.get_health_status()


# =============================================================================
# Recommendation Endpoints
# =============================================================================

@v1_router.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
@limiter.limit("30/minute")
async def recommend(
    request: Request,
    body: RecommendRequest,
    app_state: AppState = Depends(get_app_state)
):
    """
    Get anime recommendations based on liked anime.

    Provide a list of anime names you like, and the RBM model will
    suggest similar anime you might enjoy.
    """
    if not body.liked_anime:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="liked_anime must be a non-empty list")

    if app_state.anime_df is None:
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Dataset not loaded")

    matched_ids = app_state.anime_df[app_state.anime_df["name"].isin(body.liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="No matching anime found")

    try:
        top_n = min(body.top_n, 50)
        exclude_ids = body.exclude_ids[:200]
        logger.info(f"Generating {top_n} recommendations for {len(matched_ids)} matched anime (excluding {len(exclude_ids)} IDs)")

        input_vec = torch.FloatTensor([[1 if a in matched_ids else 0 for a in app_state.anime_ids]]).to(app_state.device)
        recs = get_recommendations(
            input_vec.squeeze(0),
            app_state.rbm,
            app_state.anime_ids,
            app_state.anime_df,
            top_n=top_n,
            exclude_ids=exclude_ids,
            device=app_state.device
        )
        logger.info(f"Got {len(recs)} recommendations")

        recommendations = [_build_anime_result(row, app_state) for _, row in recs.iterrows()]

        return RecommendResponse(recommendations=recommendations)

    except Exception:
        logger.exception(f"Error generating recommendations for: {body.liked_anime}")
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while generating recommendations")


@v1_router.get("/search-anime", response_model=SearchResponse, tags=["Search"])
@limiter.limit("60/minute")
async def search_anime(
    request: Request,
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    app_state: AppState = Depends(get_app_state)
):
    """
    Search for anime by name.

    Searches across name, title_english, and title_japanese fields.
    Results are paginated with limit and offset parameters.
    """
    query = (query or "").strip()

    # Validate pagination parameters
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Offset must be non-negative")

    # Empty query returns empty results
    if not query:
        return SearchResponse(results=[], total=0, limit=limit, offset=offset)

    # Validate query
    if len(query) > 100:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Query too long (max 100 chars)")

    dangerous_chars = ['<', '>', '"', "'", ';', '--', '/*', '*/']
    if any(char in query for char in dangerous_chars):
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Invalid characters in search query")

    if app_state.anime_df is None:
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Dataset not loaded")

    try:
        # Search across all name columns
        anime_df = app_state.anime_df
        name_cols = [c for c in ["name", "title_english", "title_japanese"] if c in anime_df.columns]
        mask = pd.Series(False, index=anime_df.index)
        for col in name_cols:
            mask |= anime_df[col].astype(str).str.contains(query, case=False, na=False)

        matches = anime_df[mask]
        cols = [c for c in ["anime_id", "name", "title_english", "title_japanese"] if c in matches.columns]
        matches = matches[cols].drop_duplicates()

        total_count = len(matches)
        matches = matches.iloc[offset:offset + limit]

        # Build results with metadata enrichment
        results = []
        for _, row in matches.iterrows():
            anime_id = row.get("anime_id")
            if pd.isnull(anime_id):
                continue

            try:
                anime_id = int(anime_id)
            except (TypeError, ValueError):
                continue

            results.append(_build_anime_result(row, app_state))

        return SearchResponse(results=results, total=total_count, limit=limit, offset=offset)

    except Exception:
        logger.exception(f"Error searching anime with query: {query}")
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while searching anime")


# =============================================================================
# Router Registration
# =============================================================================

v1_router.include_router(auth_router)
v1_router.include_router(favorites_router)
app.include_router(v1_router)


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
