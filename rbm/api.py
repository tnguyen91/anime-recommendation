import json
import os
import torch
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS

from constants import (
    DEFAULT_TOP_N, DEFAULT_API_HOST, DEFAULT_API_PORT,
    HTTP_BAD_REQUEST, ANIME_METADATA_FILE, CONFIG_FILE
)
from src.data_loader import load_anime_dataset
from src.model import RBM
from src.utils import preprocess_data, get_recommendations

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

app = Flask(__name__)
CORS(app)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    liked_anime = data.get("liked_anime", [])
    if not liked_anime or not isinstance(liked_anime, list):
        return jsonify({"error": "liked_anime must be a non-empty list"}), HTTP_BAD_REQUEST
    matched_ids = anime_df[anime_df["name"].isin(liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        return jsonify({"error": "No matching anime found"}), HTTP_BAD_REQUEST
    input_vec = torch.FloatTensor([[1 if a in matched_ids else 0 for a in anime_ids]]).to(device)
    recs = get_recommendations(input_vec.squeeze(0), rbm, anime_ids, anime_df, top_n=DEFAULT_TOP_N, device=device)
    response = []
    for _, row in recs.iterrows():
        info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"])) or {}
        response.append({
            "anime_id": row["anime_id"],
            "name": row["name"],
            "image_url": info.get("image_url"),
            "genre": info.get("genres", []),
            "synopsis": info.get("synopsis", "")
        })
    return jsonify({"recommendations": response})


@app.route('/search-anime', methods=['GET'])
def search_anime():
    """Search anime metadata by (partial, case-insensitive) name / English / Japanese title.

    Returns list of matches augmented with cached metadata (image_url, genres, synopsis) when available.
    Query param: ?query=...  (empty query -> empty results)
    """
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"results": []})
    if len(query) > 100:
        return jsonify({"error": "Query too long (max 100 chars)"}), HTTP_BAD_REQUEST
    name_cols = [c for c in ["name", "title_english", "title_japanese"] if c in anime_df.columns]
    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)
    matches = anime_df[mask]
    # Select core columns if present
    cols = [c for c in ["anime_id", "name", "title_english", "title_japanese"] if c in matches.columns]
    matches = matches[cols].drop_duplicates()
    results = []
    for _, row in matches.iterrows():
        info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"])) or {}
        results.append({
            "anime_id": int(row["anime_id"]),
            "name": row.get("name", ""),
            "title_english": row.get("title_english") or "",
            "title_japanese": row.get("title_japanese") or "",
            "image_url": info.get("image_url"),
            "genre": info.get("genres", []),
            "synopsis": info.get("synopsis", "")
        })
    return jsonify({"results": results})


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, debug=True)
