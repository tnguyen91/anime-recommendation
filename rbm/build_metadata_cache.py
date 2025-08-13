import json
import time
import requests

from constants import ANIME_METADATA_FILE, API_DELAY_SECONDS
from src.data_loader import load_anime_dataset

OUTPUT_FILE = ANIME_METADATA_FILE
DELAY = API_DELAY_SECONDS


def fetch_anime_info(anime_id):
    try:
        res = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}")
        res.raise_for_status()
        data = res.json()
        if "data" not in data:
            return None
        anime = data["data"]
        return {
            "image_url": anime["images"]["jpg"]["image_url"],
            "genres": [g["name"] for g in anime.get("genres", [])],
            "synopsis": anime.get("synopsis", "")
        }
    except Exception:
        return None


def build_cache():
    ratings, anime_df = load_anime_dataset()
    anime_ids = anime_df["anime_id"].dropna().unique()
    cache = {}
    for anime_id in anime_ids:
        info = fetch_anime_info(anime_id)
        if info:
            cache[int(anime_id)] = info
        time.sleep(DELAY)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved metadata cache to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_cache()
