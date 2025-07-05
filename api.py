from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, get_recommendations, make_train_test_split
from src.model import RBM
import yaml
import pandas as pd

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
    liked_titles = data.get("liked_anime", [])

    # Match titles to MAL_IDs
    matched_ids = anime_df[anime_df["Name"].isin(liked_titles)]["MAL_ID"].tolist()
    if not matched_ids:
        return jsonify({"error": "No matching anime found"}), 400

    # Convert to input vector
    input_vec = torch.FloatTensor([make_input_vector(matched_ids, anime_ids)]).to(device)

    # Get recommendations
    recommendations = get_recommendations(
        input_vec.squeeze(0),
        rbm,
        anime_ids,
        anime_df,
        top_n=10,
        device=device
    )

    return jsonify({"recommendations": recommendations.to_dict(orient="records")})

@app.route('/search-anime', methods=['GET'])
def search_anime_titles():
    query = request.args.get('query', '').strip()

    if not query:
        return jsonify({"results": []})

    name_cols = ["Name", "English name", "Japanese name"]

    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)

    matches = anime_df[mask][["MAL_ID", "Name", "English name"] + name_cols].dropna()

    # Format result for frontend (we only return 'Name' in the UI for now)
    results = matches.to_dict(orient="records")

    return jsonify({"results": results})
    
if __name__ == "__main__":
    app.run(port=5000)