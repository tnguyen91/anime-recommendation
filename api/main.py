import json
import sys
from pathlib import Path

import pandas as pd
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rbm.constants import (
    CONFIG_FILE,
    DEFAULT_TOP_N,
    HTTP_BAD_REQUEST,
    ANIME_METADATA_FILE,
)
from rbm.src.data_loader import load_anime_dataset
from rbm.src.model import RBM
from rbm.src.utils import preprocess_data, get_recommendations

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

RBM_DIR = PROJECT_ROOT / "rbm"

try:
    with open(ANIME_METADATA_FILE, "r") as f:
        anime_metadata = json.load(f)
except Exception:
    anime_metadata = {}

config_path = RBM_DIR / CONFIG_FILE
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_cfg = config["model"]
data_cfg = config["data"]
path_cfg = config["paths"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ratings = None
anime_df = None
user_anime = None
anime_ids = None
rbm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ratings, anime_df, user_anime, anime_ids, rbm

    ratings, anime_df = load_anime_dataset()

    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=data_cfg["min_likes_user"],
        min_likes_anime=data_cfg["min_likes_anime"]
    )
    anime_ids = list(user_anime.columns)

    rbm = RBM(n_visible=len(anime_ids), n_hidden=model_cfg["n_hidden"]).to(device)
    model_path = (RBM_DIR / path_cfg["model_path"]).resolve()
    if model_path.exists():
        rbm.load_state_dict(torch.load(model_path, map_location=device))

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
        raise HTTPException(status_code=500, detail="Internal server error while generating recommendations")

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
        raise HTTPException(status_code=500, detail="Internal server error while searching anime")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )