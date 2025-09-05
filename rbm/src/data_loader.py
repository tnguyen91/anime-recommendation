import glob
import os
import kagglehub
import pandas as pd

from constants import KAGGLE_DATASET, DATA_VERSIONS_PATH, DATA_DIR
from src.utils import filter_hentai

os.environ.setdefault("KAGGLEHUB_CACHE", DATA_DIR)


def _resolve_latest_dataset_version():
    """Return the absolute path to the latest downloaded dataset version folder."""
    candidates = sorted(glob.glob(DATA_VERSIONS_PATH))
    if not candidates:
        raise FileNotFoundError(
            f"No dataset versions found matching pattern: {DATA_VERSIONS_PATH}. "
            "Make sure the Kaggle dataset has been downloaded."
        )
    return candidates[-1].replace("\\", "/")


def load_anime_dataset():
    kagglehub.dataset_download(KAGGLE_DATASET)
    latest_version = _resolve_latest_dataset_version()
    rating_path = os.path.join(latest_version, "User-AnimeReview.csv")
    anime_path = os.path.join(latest_version, "Anime.csv")
    if not os.path.exists(rating_path) or not os.path.exists(anime_path):
        raise FileNotFoundError(
            f"Expected dataset files not found in {latest_version}: "
            f"{os.path.basename(rating_path)}, {os.path.basename(anime_path)}"
        )
    ratings = pd.read_csv(rating_path)
    anime = pd.read_csv(anime_path)
    return filter_hentai(ratings, anime)
