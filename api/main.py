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
import asyncio
import logging

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

cache_env = os.getenv("CACHE_DIR")
if not cache_env:
    raise EnvironmentError("CACHE_DIR environment variable must be set to a valid directory path.")
CACHE_DIR = Path(cache_env).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_URI = os.getenv("MODEL_URI")
METADATA_URI = os.getenv("METADATA_URI")

anime_metadata: dict[str, Any] = {}

model_loaded = False
metadata_loaded = False
dataset_loaded = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ratings = None
anime_df = None
user_anime = None
anime_ids = None
rbm = None

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ratings, anime_df, user_anime, anime_ids, rbm, anime_metadata
    global model_loaded, metadata_loaded, dataset_loaded

    logging.info("Starting application lifespan: running blocking startup tasks")

    anime_cache = None
    reviews_cache = None
    try:
        for p in CACHE_DIR.iterdir():
            if p.name.lower().endswith("anime.csv"):
                anime_cache = p
            if p.name.lower().endswith("user-animereview.csv") or p.name.lower().endswith("user_animereview.csv"):
                reviews_cache = p

        if anime_cache and reviews_cache:
            logging.info("Loading dataset from cache: %s, %s", anime_cache, reviews_cache)
            anime_df = pd.read_csv(anime_cache)
            ratings = pd.read_csv(reviews_cache)
        else:
            anime_uri = os.environ.get("ANIME_CSV_URI")
            review_uri = os.environ.get("USER_REVIEW_CSV_URI")
            if not anime_uri or not review_uri:
                raise RuntimeError("ANIME_CSV_URI and USER_REVIEW_CSV_URI must be set for foreground download")

            logging.info("Downloading dataset synchronously")
            anime_path = download_to_cache(anime_uri)
            review_path = download_to_cache(review_uri)
            anime_df = pd.read_csv(anime_path)
            ratings = pd.read_csv(review_path)

        try:
            user_anime, _ = preprocess_data(ratings, min_likes_user=MIN_LIKES_USER, min_likes_anime=MIN_LIKES_ANIME)
        except Exception:
            ratings_filtered, anime_filtered = preprocess_data(ratings, min_likes_user=MIN_LIKES_USER, min_likes_anime=MIN_LIKES_ANIME)
            user_anime = ratings_filtered.pivot_table(index='user_id', columns='anime_id', values='liked', fill_value=0)

        anime_ids = list(user_anime.columns) if user_anime is not None else []
        rbm = RBM(n_visible=len(anime_ids), n_hidden=N_HIDDEN).to(device) if anime_ids else None
        dataset_loaded = True
        logging.info("Dataset loaded: anime=%s users=%s", len(anime_df) if anime_df is not None else None, len(user_anime) if user_anime is not None else None)

    except Exception as e:
        logging.exception("Failed during dataset startup: %s", e)
        raise

    try:
        if MODEL_URI and (MODEL_URI.startswith("http://") or MODEL_URI.startswith("https://")):
            logging.info("Downloading model synchronously from %s", MODEL_URI)
            model_path = download_to_cache(MODEL_URI)
            if model_path.exists() and rbm is not None:
                ckpt = torch.load(model_path, map_location=device)
                state_dict = None
                if isinstance(ckpt, dict):
                    state_dict = ckpt if all(isinstance(v, torch.Tensor) for v in ckpt.values()) else ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
                else:
                    state_dict = ckpt

                EXPECTED_N_VISIBLE = len(anime_ids) if anime_ids is not None else None
                ckpt_v_bias = None
                ckpt_W = None
                if isinstance(state_dict, dict):
                    for k, v in state_dict.items():
                        if k.endswith('v_bias') and hasattr(v, 'shape'):
                            ckpt_v_bias = v.shape
                        if k.endswith('W') and hasattr(v, 'shape'):
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
                rbm.load_state_dict(state_dict)
                model_loaded = True
                logging.info("Model loaded into RBM")
            else:
                raise RuntimeError(f"Model path {model_path} does not exist or RBM not created; cannot load model.")
        else:
            raise RuntimeError("MODEL_URI not provided or not an http(s) URL; cannot load model in foreground mode.")
    except Exception as e:
        logging.exception("Failed during model startup: %s", e)
        raise

    try:
        if METADATA_URI and (METADATA_URI.startswith("http://") or METADATA_URI.startswith("https://")):
            logging.info("Downloading metadata synchronously from %s", METADATA_URI)
            metadata_path = download_to_cache(METADATA_URI)
            if metadata_path.exists():
                with metadata_path.open('r', encoding='utf-8') as f:
                    anime_metadata = json.load(f)
                metadata_loaded = True
                logging.info("Metadata loaded")
            else:
                raise RuntimeError(f"Metadata path {metadata_path} does not exist after download.")
        else:
            raise RuntimeError("METADATA_URI not provided or not an http(s) URL; cannot load metadata in foreground mode.")
    except Exception as e:
        logging.exception("Failed during metadata startup: %s", e)
        raise

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

    try:
        health_status["services"]["background"] = {
            "model_loaded": bool(model_loaded),
            "metadata_loaded": bool(metadata_loaded),
            "dataset_loaded": bool(dataset_loaded),
        }
        if not (model_loaded and metadata_loaded and dataset_loaded):
            health_status["status"] = "degraded"
    except Exception:
        pass

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
