import json
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Any
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

from api.config import (
    DEFAULT_TOP_N,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_ERROR,
    DEFAULT_MODEL_PATH,
    DEFAULT_METADATA_PATH,
    CACHE_DIR_DEFAULT,
    MIN_LIKES_USER,
    MIN_LIKES_ANIME,
    N_HIDDEN,
)
from api.inference.data_loader import load_anime_dataset
from api.inference.model import RBM
from api.inference.preprocess import preprocess_data
from api.inference.recommender import get_recommendations

class RecommendRequest(BaseModel):
    liked_anime: list[str]

class Recommendation(BaseModel):
    anime_id: int
    name: str
    image_url: str | None
    genre: list[str]
    synopsis: str | None

class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]

class SearchResult(BaseModel):
    anime_id: int
    name: str
    title_english: str | None
    title_japanese: str | None
    image_url: str | None
    genre: list[str]
    synopsis: str | None

class SearchResponse(BaseModel):
    results: list[SearchResult]

CACHE_DIR = Path(os.getenv("CACHE_DIR", str(CACHE_DIR_DEFAULT))).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_URI = os.getenv("MODEL_URI")
METADATA_URI = os.getenv("METADATA_URI")

anime_metadata: dict[str, Any] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _resolve_resource(uri: str | None, default_path: Path) -> Path:
    """
    Resolve a resource URI to a local Path.

    Supported inputs:
    - None -> use default_path
    - local filesystem path (absolute or relative to project root)
    - http(s) URLs (not downloaded automatically; raise instructive error)

    S3 (s3://) and other cloud-specific schemes are intentionally not supported here.
    If you previously relied on S3, upload the files into the project `data/` folder or
    provide a local path via environment variables.
    """

    if uri:
        # Reject cloud-specific URIs on purpose to keep repo AWS-free
        if uri.startswith("s3://"):
            raise ValueError("S3 URIs are not supported in this workspace. Please provide a local path to the model and metadata.")

        # http(s) handled explicitly but not downloaded automatically
        if uri.startswith("http://") or uri.startswith("https://"):
            raise ValueError("HTTP URIs are not automatically downloaded. Please provide a local file path or add download logic in your deployment step.")

        candidate = Path(uri)
        path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    else:
        path = default_path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resource not found at {path}. Provide a valid local path.")

    return path

ratings = None
anime_df = None
user_anime = None
anime_ids = None
rbm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ratings, anime_df, user_anime, anime_ids, rbm, anime_metadata

    ratings, anime_df = load_anime_dataset()

    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=MIN_LIKES_USER,
        min_likes_anime=MIN_LIKES_ANIME
    )
    anime_ids = list(user_anime.columns)

    rbm = RBM(n_visible=len(anime_ids), n_hidden=N_HIDDEN).to(device)

    model_path = _resolve_resource(MODEL_URI, DEFAULT_MODEL_PATH)
    if model_path.exists():
        rbm.load_state_dict(torch.load(model_path, map_location=device))

    metadata_path = _resolve_resource(METADATA_URI, DEFAULT_METADATA_PATH)
    if metadata_path.exists():
        global anime_metadata
        with metadata_path.open("r", encoding="utf-8") as f:
            anime_metadata = json.load(f)

    yield

app = FastAPI(
    title="Anime Recommendation API",
    version="1.0.0",
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    health_status = {
        "status": "ok",
        "version": "1.0.0",
        "services": {}
    }

    try:
        if rbm is not None:
            health_status["services"]["model"] = {"status": "ok", "type": "RBM"}
        else:
            health_status["services"]["model"] = {"status": "error", "error": "Model not loaded"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["model"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    try:
        if anime_df is not None and len(anime_df) > 0:
            health_status["services"]["data"] = {
                "status": "ok",
                "anime_count": len(anime_df),
                "user_count": len(user_anime) if user_anime is not None else 0
            }
        else:
            health_status["services"]["data"] = {"status": "error", "error": "Data not loaded"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["data"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"

    return health_status

@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(request: RecommendRequest):
    liked_anime = request.liked_anime

    if not liked_anime:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="liked_anime must be a non-empty list")

    matched_ids = anime_df[anime_df["name"].isin(liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="No matching anime found")

    try:
        input_vec = torch.FloatTensor([[1 if a in matched_ids else 0 for a in anime_ids]]).to(device)
        recs = get_recommendations(input_vec.squeeze(0), rbm, anime_ids, anime_df, top_n=DEFAULT_TOP_N, device=device)
        response = []
        for _, row in recs.iterrows():
            info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"])) or {}
            response.append(Recommendation(
                anime_id=int(row["anime_id"]),
                name=row["name"],
                image_url=info.get("image_url"),
                genre=info.get("genres", []),
                synopsis=info.get("synopsis")
            ))

        result = RecommendResponse(recommendations=response)
        return result

    except Exception as e:
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while generating recommendations")

@app.get("/search-anime", response_model=SearchResponse, tags=["Search"])
async def search_anime(query: str = ""):
    query = (query or "").strip()

    if not query:
        return SearchResponse(results=[])

    if len(query) > 100:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Query too long (max 100 chars)")

    dangerous_chars = ['<', '>', '"', "'", ';', '--', '/*', '*/']
    if any(char in query for char in dangerous_chars):
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Invalid characters in search query")

    try:
        name_cols = [c for c in ["name", "title_english", "title_japanese"] if c in anime_df.columns]
        mask = pd.Series(False, index=anime_df.index)
        for col in name_cols:
            mask |= anime_df[col].astype(str).str.contains(query, case=False, na=False)
        matches = anime_df[mask]
        cols = [c for c in ["anime_id", "name", "title_english", "title_japanese"] if c in matches.columns]
        matches = matches[cols].drop_duplicates()
        results = []
        for _, row in matches.iterrows():
            raw_id = row.get("anime_id")
            if pd.isnull(raw_id):
                continue

            try:
                normalized_id = int(raw_id)
            except (TypeError, ValueError):
                continue

            metadata_info = anime_metadata.get(str(normalized_id)) or anime_metadata.get(normalized_id, {})

            results.append(SearchResult(
                anime_id=normalized_id,
                name="" if pd.isnull(row.get("name")) else row.get("name", ""),
                title_english="" if pd.isnull(row.get("title_english")) else row.get("title_english", ""),
                title_japanese="" if pd.isnull(row.get("title_japanese")) else row.get("title_japanese", ""),
                image_url=metadata_info.get("image_url"),
                genre=metadata_info.get("genres", []),
                synopsis=metadata_info.get("synopsis")
            ))

        return SearchResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=HTTP_INTERNAL_ERROR, detail="Internal server error while searching anime")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
