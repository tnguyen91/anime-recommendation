import json
import time

import requests
import torch
import yaml
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.data_loader import load_anime_dataset
from src.model import RBM
from src.utils import preprocess_data, get_recommendations, make_train_test_split
with open("data/anime_metadata.json", "r") as f:
    anime_metadata = json.load(f)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config["paths"]["model_path"]

# Load dataset
ratings, anime_df = load_anime_dataset()
user_anime, _ = preprocess_data(
    ratings,
    min_likes_user=config["data"]["min_likes_user"],
    min_likes_anime=config["data"]["min_likes_anime"]
)

anime_ids = list(user_anime.columns)

# Load trained model
rbm = RBM(n_visible=len(anime_ids), n_hidden=config["model"]["n_hidden"]).to(device)
rbm = torch.quantization.quantize_dynamic(rbm, {torch.nn.Linear}, dtype=torch.qint8)
rbm.load_state_dict(torch.load(model_path, map_location=device))
rbm.eval()

# Start Flask app
app = Flask(__name__)
CORS(app)

def make_input_vector(liked_anime_ids, anime_ids):
    return [1 if anime_id in liked_anime_ids else 0 for anime_id in anime_ids]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    liked_anime = data.get("liked_anime", [])

    matched_ids = anime_df[anime_df["name"].isin(liked_anime)]["anime_id"].tolist()
    if not matched_ids:
        return jsonify({"error": "No matching anime found"}), 400

    input_vec = torch.FloatTensor([make_input_vector(matched_ids, anime_ids)]).to(device)

    recs = get_recommendations(
        input_vec.squeeze(0),
        rbm,
        anime_ids,
        anime_df,
        top_n=10,
        device=device
    )

    # Add image and genre info
    recommendations = []
    for _, row in recs.iterrows():
        #image_url, genre, synopsis = fetch_anime_info(row["anime_id"])
        info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"]))
        image_url = info.get("image_url") if info else None
        genre = info.get("genres", []) if info else []
        synopsis = info.get("synopsis", "") if info else ""
        recommendations.append({
            "anime_id": row["anime_id"],
            "name": row["name"],
            "title_english": row.get("title_english", ""),
            "image_url": image_url,
            "genre": genre,
            "synopsis": synopsis
        })

    return jsonify({"recommendations": recommendations})

def fetch_anime_info(anime_id):
    try:
        res = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}")
        res.raise_for_status()
        data = res.json()
        anime = data["data"]
        image_url = anime["images"]["jpg"]["image_url"]
        genres = [g["name"] for g in anime.get("genres", [])]
        synopsis = anime.get("synopsis", "")
        return image_url, genres, synopsis
    except Exception as e:
        print(f"Error fetching info for ID {anime_id}: {e}")
        return None, [], ""

@app.route('/search-anime', methods=['GET'])
def search_anime():
    query = request.args.get('query', '').strip()

    if not query:
        return jsonify({"results": []})

    name_cols = ["name", "title_english", "title_japanese"]

    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)

    matches = anime_df[mask][["anime_id", "name", "title_english"]].dropna()

    results = []
    for _, row in matches.iterrows():
        #image_url, genre, synopsis = fetch_anime_info(row["anime_id"])
        info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"]))
        image_url = info.get("image_url") if info else None
        genre = info.get("genres", []) if info else []
        synopsis = info.get("synopsis", "") if info else ""
        results.append({
            "anime_id": row["anime_id"],
            "name": row["name"],
            "title_english": row["title_english"],
            "image_url": image_url,
            "genre": genre,
            "synopsis": synopsis
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
