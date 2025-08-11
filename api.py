import json
import os
import time

import pandas as pd
import requests
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
from src.utils import preprocess_data, get_recommendations, make_train_test_split

try:
    with open(ANIME_METADATA_FILE, "r") as f:
        anime_metadata = json.load(f)
except FileNotFoundError:
    print(f"Warning: Metadata file '{ANIME_METADATA_FILE}' not found. Image/genre info won't be available.")
    anime_metadata = {}
except json.JSONDecodeError as e:
    print(f"Error parsing metadata JSON: {e}")
    anime_metadata = {}

try:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = config["paths"]["model_path"]

try:
    ratings, anime_df = load_anime_dataset()
    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=config["data"]["min_likes_user"],
        min_likes_anime=config["data"]["min_likes_anime"]
    )

    anime_ids = list(user_anime.columns)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    rbm = RBM(n_visible=len(anime_ids), n_hidden=config["model"]["n_hidden"]).to(device)
    rbm = torch.quantization.quantize_dynamic(rbm, {torch.nn.Linear}, dtype=torch.qint8)
    rbm.load_state_dict(torch.load(model_path, map_location=device))
    rbm.eval()
    
except Exception as e:
    print(f"Error initializing API: {e}")
    print("Make sure the model is trained and data is available.")
    exit(1)

# Start Flask app
app = Flask(__name__)
CORS(app)

def make_input_vector(liked_anime_ids, anime_ids):
    return [1 if anime_id in liked_anime_ids else 0 for anime_id in anime_ids]

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Generate anime recommendations based on user's liked anime.
    
    API endpoint that accepts a list of anime names, converts them to 
    a binary preference vector, and returns top-N recommendations using 
    the trained RBM model.
    
    Request JSON:
        {
            "liked_anime": ["Anime Name 1", "Anime Name 2", ...]
        }
        
    Returns:
        JSON response with recommendations or error message:
        {
            "recommendations": [
                {"anime_id": int, "name": str, "score": float}, ...
            ]
        }
        
    Status Codes:
        200: Success with recommendations
        400: Bad request (invalid input)
        500: Internal server error
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), HTTP_BAD_REQUEST
        
        liked_anime = data.get("liked_anime", [])
        if not liked_anime:
            return jsonify({"error": "No liked anime provided"}), HTTP_BAD_REQUEST
        
        if not isinstance(liked_anime, list):
            return jsonify({"error": "liked_anime must be a list"}), HTTP_BAD_REQUEST

        matched_ids = anime_df[anime_df["name"].isin(liked_anime)]["anime_id"].tolist()
        if not matched_ids:
            return jsonify({"error": "No matching anime found in database"}), HTTP_BAD_REQUEST

        input_vec = torch.FloatTensor([make_input_vector(matched_ids, anime_ids)]).to(device)
        recs = get_recommendations(
            input_vec.squeeze(0),
            rbm,
            anime_ids,
            anime_df,
            top_n=DEFAULT_TOP_N,
            device=device
        )

        recommendations = []
        for _, row in recs.iterrows():
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
        
    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

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
    """
    Search for anime by name across multiple title fields.
    
    API endpoint that searches anime database using case-insensitive 
    substring matching across name, English title, and Japanese title fields.
    
    Query Parameters:
        q (str): Search query string (required)
        
    Returns:
        JSON response with matching anime:
        {
            "results": [
                {"anime_id": int, "name": str, "title_english": str, "title_japanese": str}, ...
            ]
        }
        
    Status Codes:
        200: Success with search results (may be empty list)
        400: Bad request (missing query parameter)
        500: Internal server error
    """
    try:
        query = request.args.get('query', '').strip()

        if not query:
            return jsonify({"results": []})
        
        if len(query) > 100:
            return jsonify({"error": "Query too long (max 100 characters)"}), HTTP_BAD_REQUEST

        name_cols = ["name", "title_english", "title_japanese"]
        mask = False
        for col in name_cols:
            mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)

        matches = anime_df[mask][["anime_id", "name", "title_english"]].dropna()

        results = []
        for _, row in matches.iterrows():
            info = anime_metadata.get(str(row["anime_id"])) or anime_metadata.get(int(row["anime_id"]))
            image_url = info.get("image_url") if info else None
            genre = info.get("genres", []) if info else []
            synopsis = info.get("synopsis", "") if info else ""
            
            results.append({
                "anime_id": int(row["anime_id"]),
                "name": row["name"],
                "title_english": row["title_english"] if pd.notna(row["title_english"]) else "",
                "image_url": image_url,
                "genre": genre,
                "synopsis": synopsis
            })

        return jsonify({"results": results})
        
    except Exception as e:
        print(f"Error in search endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, debug=True)
