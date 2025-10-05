import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

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

class RecommendRequest(BaseModel):
    liked_anime: list[str]

class AnimeResult(BaseModel):
    anime_id: int
    name: str
    title_english: str | None
    title_japanese: str | None
    image_url: str | None
    genre: list[str]
    synopsis: str | None

class RecommendResponse(BaseModel):
    recommendations: list[AnimeResult]

class SearchResponse(BaseModel):
    results: list[AnimeResult]

cache_env = os.getenv("CACHE_DIR")
if cache_env:
    CACHE_DIR = Path(cache_env).resolve()
else:
    CACHE_DIR = Path("/tmp/cache").resolve()

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_URI = os.getenv("MODEL_URI")
METADATA_URI = os.getenv("METADATA_URI")

anime_metadata: dict[str, Any] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ratings_df = None
anime_df = None
user_anime = None
anime_ids = None
rbm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ratings_df, anime_df, user_anime, anime_ids, rbm, anime_metadata
    try:
        ratings_df, anime_df = load_anime_dataset()
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    try:
        active_anime, active_ratings = filter_data(
            ratings_df,
            min_likes_user=MIN_LIKES_USER,
            min_likes_anime=MIN_LIKES_ANIME
        )
        anime_ids = list(active_anime) if active_anime is not None else []
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess data: {e}")

    try:
        rbm = RBM(n_visible=len(anime_ids), n_hidden=N_HIDDEN).to(device) if anime_ids else None
    except Exception as e:
        raise RuntimeError(f"Failed to create RBM instance: {e}")

    try:
        if MODEL_URI:
            try:
                model_path = download_to_cache(MODEL_URI)
                if model_path.exists() and rbm is not None:
                    try:
                        ckpt = torch.load(model_path, map_location=device)
                        state_dict = ckpt if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()) else ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
                        EXPECTED_N_VISIBLE = len(anime_ids) if anime_ids is not None else None

                        ckpt_v_bias = None
                        ckpt_W = None
                        for k, v in state_dict.items():
                            if k.endswith('v_bias') and isinstance(v, torch.Tensor):
                                ckpt_v_bias = v.shape
                            if k.endswith('W') and isinstance(v, torch.Tensor):
                                ckpt_W = v.shape

                        mismatch = False
                        if ckpt_v_bias is not None and EXPECTED_N_VISIBLE is not None:
                            if ckpt_v_bias[0] != EXPECTED_N_VISIBLE:
                                mismatch = True
                        if ckpt_W is not None and EXPECTED_N_VISIBLE is not None:
                            if ckpt_W[1] != EXPECTED_N_VISIBLE:
                                mismatch = True

                        if mismatch:
                            raise RuntimeError("Checkpoint shape mismatch: RBM will be uninitialized.")
                        else:
                            rbm.load_state_dict(state_dict)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load model checkpoint contents: {e}")
                else:
                    raise FileNotFoundError(f"Model path {model_path} does not exist or RBM not created; starting without model.")
            except Exception as e:
                raise RuntimeError(f"Failed to download/load model from {MODEL_URI}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_URI}: {e}")

    try:
        if METADATA_URI:
            try:
                metadata_path = download_to_cache(METADATA_URI)
                if metadata_path.exists():
                    with metadata_path.open("r", encoding="utf-8") as f:
                        anime_metadata = json.load(f)
                else:
                    raise FileNotFoundError(f"Metadata path {metadata_path} does not exist after download.")
            except Exception as e:
                raise RuntimeError(f"Failed to download/load metadata from {METADATA_URI}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from {METADATA_URI}: {e}")

    try:
        anime_cache = None
        reviews_cache = None
        for p in CACHE_DIR.iterdir():
            if p.name.lower().endswith("anime.csv"):
                anime_cache = str(p)
            if p.name.lower().endswith("user-animereview.csv") or p.name.lower().endswith("user_animereview.csv"):
                reviews_cache = str(p)

        if anime_cache or reviews_cache:
            try:
                if anime_cache:
                    anime_df = pd.read_csv(anime_cache)
            except Exception as e:
                print(f"Warning: failed to load cached data: {e}")
    except Exception as e:
        print(f"Warning: failed to load cache directory: {e}")

    yield

app = FastAPI(
    title="Anime Recommendation API",
    version="1.0.0",
    lifespan=lifespan 
)

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

@app.get("/")
async def root():
    return await health()

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
            health_status["services"]["dataset"] = {
                "status": "ok",
                "anime_count": len(anime_df),
                "rating_count": len(ratings_df) 
            }
        else:
            health_status["services"]["dataset"] = {"status": "error", "error": "Dataset not loaded"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["dataset"] = {"status": "error", "error": str(e)}
        health_status["status"] = "degraded"
    
    try:
        if anime_metadata is not None and len(anime_metadata) > 0:
            health_status["services"]["metadata"] = {"status": "ok", "entry_count": len(anime_metadata)}
        else:
            health_status["services"]["metadata"] = {"status": "error", "error": "Metadata not loaded"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["metadata"] = {"status": "error", "error": str(e)}
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
            response.append(AnimeResult(
                anime_id=int(row["anime_id"]),
                name=row["name"],
                title_english="" if pd.isnull(row.get("title_english")) else row.get("title_english", ""),
                title_japanese="" if pd.isnull(row.get("title_japanese")) else row.get("title_japanese", ""),
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

            results.append(AnimeResult(
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
