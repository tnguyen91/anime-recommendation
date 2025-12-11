"""
Anime Recommendation API - Main Application.

This is the entry point for the FastAPI application. It:
1. Loads the RBM model and anime datasets at startup
2. Configures middleware (CORS, rate limiting, logging)
3. Mounts API routers for recommendations, auth, and favorites

Run locally with: uvicorn api.main:app --reload
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.config import (
    DEFAULT_TOP_N,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_ERROR,
    MIN_LIKES_USER,
    MIN_LIKES_ANIME,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

# Request/Response schemas for recommendation endpoints
class RecommendRequest(BaseModel):
    """Request body for getting recommendations."""
    liked_anime: list[str]
    top_n: int = DEFAULT_TOP_N


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


# Configuration from environment
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URI = os.getenv("MODEL_URI")
METADATA_URI = os.getenv("METADATA_URI")

# Global state - loaded at startup
anime_metadata: dict[str, Any] = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ratings_df = None
anime_df = None
user_anime = None
anime_ids = None
rbm = None


def _load_model(model_path: Path, expected_n_visible: int):
    """Load RBM model from checkpoint file."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    
    Loads datasets, model, and metadata at startup.
    """
    global ratings_df, anime_df, user_anime, anime_ids, rbm, anime_metadata
    
    # Load anime dataset
    logger.info("Loading anime dataset...")
    ratings_df, anime_df = load_anime_dataset()
    
    # Filter data for active users and anime
    logger.info("Filtering data...")
    active_anime, _ = filter_data(
        ratings_df,
        min_likes_user=MIN_LIKES_USER,
        min_likes_anime=MIN_LIKES_ANIME
    )
    anime_ids = list(active_anime) if active_anime is not None else []
    
    # Initialize RBM model
    if anime_ids:
        logger.info(f"Initializing RBM with {len(anime_ids)} visible units...")
        rbm = RBM(n_visible=len(anime_ids), n_hidden=N_HIDDEN).to(device)
        
        # Load pretrained weights if available
        if MODEL_URI:
            logger.info(f"Loading model from {MODEL_URI}...")
            model_path = download_to_cache(MODEL_URI)
            if model_path.exists():
                state_dict = _load_model(model_path, len(anime_ids))
                rbm.load_state_dict(state_dict)
                logger.info("Model loaded successfully")
    
    # Load anime metadata
    if METADATA_URI:
        logger.info(f"Loading metadata from {METADATA_URI}...")
        metadata_path = download_to_cache(METADATA_URI)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                anime_metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(anime_metadata)} anime")
    
    # Load cached CSV data if available
    for p in CACHE_DIR.iterdir():
        if p.name.lower().endswith("anime.csv"):
            anime_df = pd.read_csv(str(p))
            logger.info(f"Loaded anime data from cache: {p.name}")
            break
    
    # Share metadata with favorites module
    set_anime_cache(anime_metadata, anime_df)
    
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

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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

@app.get("/")
async def root():
    """Redirect to health check."""
    return await health()


@app.get("/health")
async def health():
    """
    Health check endpoint for monitoring.
    
    Returns status of model, dataset, and metadata services.
    """
    health_status = {
        "status": "ok",
        "version": "1.0.0",
        "services": {}
    }

    # Check model
    if rbm is not None:
        health_status["services"]["model"] = {"status": "ok", "type": "RBM"}
    else:
        health_status["services"]["model"] = {"status": "error", "error": "Model not loaded"}
        health_status["status"] = "degraded"

    # Check dataset
    if anime_df is not None and len(anime_df) > 0:
        health_status["services"]["dataset"] = {
            "status": "ok",
            "anime_count": len(anime_df),
            "rating_count": len(ratings_df) if ratings_df is not None else 0
        }
    else:
        health_status["services"]["dataset"] = {"status": "error", "error": "Dataset not loaded"}
        health_status["status"] = "degraded"

    # Check metadata
    if anime_metadata and len(anime_metadata) > 0:
        health_status["services"]["metadata"] = {"status": "ok", "entry_count": len(anime_metadata)}
    else:
        health_status["services"]["metadata"] = {"status": "error", "error": "Metadata not loaded"}
        health_status["status"] = "degraded"

    return health_status


@v1_router.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
@limiter.limit("30/minute")
async def recommend(request: Request, body: RecommendRequest):
    """
    Get anime recommendations based on liked anime.
    
    Provide a list of anime names you like, and the RBM model will
    suggest similar anime you might enjoy.
    """
    if not body.liked_anime:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="liked_anime must be a non-empty list")

    matched_ids = anime_df[anime_df["name"].isin(body.liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="No matching anime found")

    try:
        top_n = min(body.top_n, 50)
        logger.info(f"Generating {top_n} recommendations for {len(matched_ids)} matched anime")
        input_vec = torch.FloatTensor([[1 if a in matched_ids else 0 for a in anime_ids]]).to(device)
        recs = get_recommendations(input_vec.squeeze(0), rbm, anime_ids, anime_df, top_n=top_n, device=device)
        logger.info(f"Got {len(recs)} recommendations")
        
        recommendations = []
        for _, row in recs.iterrows():
            info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"])) or {}
            recommendations.append(AnimeResult(
                anime_id=int(row["anime_id"]),
                name=row["name"],
                title_english="" if pd.isnull(row.get("title_english")) else row.get("title_english", ""),
                title_japanese="" if pd.isnull(row.get("title_japanese")) else row.get("title_japanese", ""),
                image_url=info.get("image_url"),
                genre=info.get("genres", []),
                synopsis=info.get("synopsis")
            ))

        return RecommendResponse(recommendations=recommendations)

    except Exception:
        logger.exception(f"Error generating recommendations for: {body.liked_anime}")
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while generating recommendations")


@v1_router.get("/search-anime", response_model=SearchResponse, tags=["Search"])
@limiter.limit("60/minute")
async def search_anime(request: Request, query: str = "", limit: int = 20, offset: int = 0):
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

    try:
        # Search across all name columns
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

            metadata = anime_metadata.get(str(anime_id)) or anime_metadata.get(anime_id, {})
            results.append(AnimeResult(
                anime_id=anime_id,
                name="" if pd.isnull(row.get("name")) else row.get("name", ""),
                title_english="" if pd.isnull(row.get("title_english")) else row.get("title_english", ""),
                title_japanese="" if pd.isnull(row.get("title_japanese")) else row.get("title_japanese", ""),
                image_url=metadata.get("image_url"),
                genre=metadata.get("genres", []),
                synopsis=metadata.get("synopsis")
            ))

        return SearchResponse(results=results, total=total_count, limit=limit, offset=offset)

    except Exception:
        logger.exception(f"Error searching anime with query: {query}")
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while searching anime")


# Mount routers
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
