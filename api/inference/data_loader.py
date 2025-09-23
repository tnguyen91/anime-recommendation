from __future__ import annotations

import glob
import os
from pathlib import Path

import kagglehub
import pandas as pd

from api.config import KAGGLE_DATASET, DATA_VERSIONS_PATH, DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)
from .preprocess import filter_hentai

os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR))


def _resolve_latest_dataset_version() -> str:
    candidates = sorted(glob.glob(DATA_VERSIONS_PATH))
    if not candidates:
        raise FileNotFoundError(
            f"No dataset versions found matching pattern: {DATA_VERSIONS_PATH}. "
            "Ensure the Kaggle dataset has been downloaded."
        )
    return candidates[-1].replace("\\", "/")


def _download_dataset() -> None:
    try:
        kagglehub.dataset_download(KAGGLE_DATASET)
    except AttributeError:
        kagglehub.client.DatasetClient().download(KAGGLE_DATASET)


def load_anime_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = sorted(glob.glob(DATA_VERSIONS_PATH))
    if not candidates:
        try:
            _download_dataset()
        except Exception:
            pass

    latest_version = _resolve_latest_dataset_version()
    latest_path = Path(latest_version)
    rating_path = latest_path / "User-AnimeReview.csv"
    anime_path = latest_path / "Anime.csv"
    if not rating_path.exists() or not anime_path.exists():
        raise FileNotFoundError(
            f"Expected dataset files not found in {latest_version}: "
            f"{rating_path.name}, {anime_path.name}"
        )
    ratings = pd.read_csv(rating_path)
    anime = pd.read_csv(anime_path)
    return filter_hentai(ratings, anime)
