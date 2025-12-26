"""Anime Recommendation API powered by RBM-based collaborative filtering."""
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, APIRouter, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.app_state import AppState, init_app_state
from api.config import DEFAULT_TOP_N, MIN_LIKES_ANIME, MIN_LIKES_USER, N_HIDDEN
from api.inference.data_loader import load_anime_dataset
from api.inference.downloads import download_to_cache
from api.inference.model import RBM
from api.inference.preprocess import filter_data
from api.inference.recommender import get_recommendations
from api.auth.router import router as auth_router
from api.favorites.router import router as favorites_router
from api.settings import settings
from api.monitoring import get_prediction_logger

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])


class RecommendRequest(BaseModel):
    """Recommendation request parameters."""
    liked_anime: list[str]
    top_n: int = DEFAULT_TOP_N
    exclude_ids: list[int] = []


class AnimeResult(BaseModel):
    """Anime item in API responses."""
    anime_id: int
    name: str
    title_english: str | None
    title_japanese: str | None
    image_url: str | None
    genre: list[str]
    synopsis: str | None


class RecommendResponse(BaseModel):
    """Recommendation response."""
    recommendations: list[AnimeResult]


class SearchResponse(BaseModel):
    """Paginated search response."""
    results: list[AnimeResult]
    total: int
    limit: int
    offset: int


def _safe_str(value) -> str:
    """Convert value to string, returning empty string for null/NaN."""
    return "" if pd.isnull(value) else str(value)


def _load_model(model_path, device: torch.device) -> dict:
    """Load RBM checkpoint from file."""
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "W" in ckpt:
        return ckpt
    return ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))


def _build_anime_result(row: pd.Series, app_state: AppState) -> AnimeResult:
    """Build AnimeResult from DataFrame row with metadata enrichment."""
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


def get_app_state(request: Request) -> AppState:
    """Dependency injection for application state."""
    return request.app.state.app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application state and load ML model at startup."""
    app_state = init_app_state()
    app.state.app_state = app_state

    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading anime dataset...")
    app_state.ratings_df, app_state.anime_df = load_anime_dataset()

    logger.info("Filtering data...")
    active_anime, _ = filter_data(
        app_state.ratings_df,
        min_likes_user=MIN_LIKES_USER,
        min_likes_anime=MIN_LIKES_ANIME
    )
    app_state.anime_ids = list(active_anime) if active_anime is not None else []

    if app_state.anime_ids:
        logger.info(f"Initializing RBM with {len(app_state.anime_ids)} visible units...")
        app_state.rbm = RBM(n_visible=len(app_state.anime_ids), n_hidden=N_HIDDEN).to(app_state.device)

        if settings.model_uri:
            logger.info(f"Loading model from {settings.model_uri}...")
            model_path = download_to_cache(settings.model_uri)
            if model_path.exists():
                state_dict = _load_model(model_path, app_state.device)
                app_state.rbm.load_state_dict(state_dict)
                logger.info("Model loaded successfully")

    if settings.metadata_uri:
        logger.info(f"Loading metadata from {settings.metadata_uri}...")
        metadata_path = download_to_cache(settings.metadata_uri)
        app_state.load_metadata(metadata_path)

    for p in settings.cache_dir.iterdir():
        if p.name.lower().endswith("anime.csv"):
            app_state.load_anime_csv(p)
            logger.info(f"Loaded anime data from cache: {p.name}")
            break

    app_state.is_initialized = True
    logger.info("Startup complete")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Anime Recommendation API",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request method, path, status, and duration."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    if request.url.path not in ["/health", "/"]:
        logger.info(f"{request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration:.3f}s")

    return response


@app.get("/")
async def root():
    """Root endpoint redirects to health check."""
    return await health()


@app.get("/health")
async def health(app_state: AppState = Depends(get_app_state)):
    """Health check endpoint for monitoring."""
    return app_state.get_health_status()


@v1_router.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
@limiter.limit("30/minute")
async def recommend(
    request: Request,
    body: RecommendRequest,
    app_state: AppState = Depends(get_app_state)
):
    """Get personalized anime recommendations based on liked anime."""
    # ========== INPUT VALIDATION ==========
    if not body.liked_anime:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="liked_anime must be a non-empty list")

    if app_state.anime_df is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Dataset not loaded")

    matched_ids = app_state.anime_df[app_state.anime_df["name"].isin(body.liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No matching anime found")

    # ========== MONITORING: Start timing ==========
    # We measure how long the prediction takes (latency)
    # High latency = bad user experience
    prediction_start = time.time()
    pred_logger = get_prediction_logger()

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

        # Extract output anime IDs for logging
        output_anime_ids = recs["anime_id"].tolist() if not recs.empty else []

        # ========== MONITORING: Log successful prediction ==========
        latency_ms = (time.time() - prediction_start) * 1000
        log_entry = pred_logger.create_log_entry(
            input_anime_ids=matched_ids,
            output_anime_ids=output_anime_ids,
            latency_ms=latency_ms,
            user_id=None,  # TODO: Extract from JWT if authenticated
            success=True
        )
        pred_logger.log(log_entry)

        recommendations = [_build_anime_result(row, app_state) for _, row in recs.iterrows()]
        return RecommendResponse(recommendations=recommendations)

    except Exception as e:
        # ========== MONITORING: Log failed prediction ==========
        latency_ms = (time.time() - prediction_start) * 1000
        log_entry = pred_logger.create_log_entry(
            input_anime_ids=matched_ids,
            output_anime_ids=[],
            latency_ms=latency_ms,
            user_id=None,
            success=False,
            error_message=str(e)
        )
        pred_logger.log(log_entry)

        logger.exception(f"Error generating recommendations for: {body.liked_anime}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while generating recommendations")


@v1_router.get("/search-anime", response_model=SearchResponse, tags=["Search"])
@limiter.limit("60/minute")
async def search_anime(
    request: Request,
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    app_state: AppState = Depends(get_app_state)
):
    """Search anime by name with pagination."""
    query = (query or "").strip()

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Offset must be non-negative")

    if not query:
        return SearchResponse(results=[], total=0, limit=limit, offset=offset)

    if len(query) > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query too long (max 100 chars)")

    if app_state.anime_df is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Dataset not loaded")

    try:
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while searching anime")


v1_router.include_router(auth_router)
v1_router.include_router(favorites_router)
app.include_router(v1_router)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
