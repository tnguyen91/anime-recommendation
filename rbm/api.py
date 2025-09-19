import json
import os
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

from constants import (
    DEFAULT_TOP_N, ANIME_METADATA_FILE, CONFIG_FILE, HTTP_BAD_REQUEST
)
from src.data_loader import load_anime_dataset
from src.model import RBM
from src.utils import preprocess_data, get_recommendations

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

try:
    with open(ANIME_METADATA_FILE, "r") as f:
        anime_metadata = json.load(f)
except Exception:
    anime_metadata = {}

config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config["paths"]["model_path"] 

ratings, anime_df = load_anime_dataset()
user_anime, _ = preprocess_data(
    ratings,
    min_likes_user=config["data"]["min_likes_user"],
    min_likes_anime=config["data"]["min_likes_anime"]
)
anime_ids = list(user_anime.columns)
rbm = RBM(n_visible=len(anime_ids), n_hidden=config["model"]["n_hidden"]).to(device)
if os.path.exists(model_path):
    rbm.load_state_dict(torch.load(model_path, map_location=device))
rbm.eval()

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

app = FastAPI(title="Anime Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    liked_anime = request.liked_anime
    if not liked_anime:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="liked_anime must be a non-empty list")
    
    matched_ids = anime_df[anime_df["name"].isin(liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="No matching anime found")
    
    cache_key = f"rec:{','.join(map(str, sorted(matched_ids)))}"
    
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return RecommendResponse.parse_raw(cached_result)
    
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
    redis_client.setex(cache_key, 3600, result.json())
    
    return result


@app.get("/search-anime", response_model=SearchResponse)
async def search_anime(query: str):
    if not query:
        return SearchResponse(results=[])
    if len(query) > 100:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Query too long (max 100 chars)")
    name_cols = [c for c in ["name", "title_english", "title_japanese"] if c in anime_df.columns]
    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)
    matches = anime_df[mask]
    cols = [c for c in ["anime_id", "name", "title_english", "title_japanese"] if c in matches.columns]
    matches = matches[cols].drop_duplicates()
    results = []
    for _, row in matches.iterrows():
        info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"])) or {}
        results.append(SearchResult(
            anime_id=int(row["anime_id"]),
            name=row.get("name", ""),
            title_english=row.get("title_english") or "",
            title_japanese=row.get("title_japanese") or "",
            image_url=info.get("image_url"),
            genre=info.get("genres", []),
            synopsis=info.get("synopsis")
        ))
    return SearchResponse(results=results)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)