from __future__ import annotations
import os
from typing import Tuple
import pandas as pd
from .preprocess import filter_hentai
from .downloads import download_to_cache

def load_anime_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    anime_uri = os.environ.get("ANIME_CSV_URI")
    review_uri = os.environ.get("USER_REVIEW_CSV_URI")

    if not anime_uri or not review_uri:
        raise EnvironmentError("Both ANIME_CSV_URI and USER_REVIEW_CSV_URI environment variables must be set.")

    try:
        anime_cached = download_to_cache(anime_uri)
        review_cached = download_to_cache(review_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset CSVs to cache: {e}")

    if not anime_cached.exists() or not review_cached.exists():
        raise FileNotFoundError(f"Dataset CSVs not found in cache: {anime_cached}, {review_cached}")

    ratings = pd.read_csv(review_cached)
    anime = pd.read_csv(anime_cached)
    return filter_hentai(ratings, anime)
