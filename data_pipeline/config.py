"""Configuration settings for the data pipeline."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"

KAGGLE_DATA_DIR = RAW_DATA_DIR / "kaggle"
JIKAN_DATA_DIR = RAW_DATA_DIR / "jikan"
APP_LOGS_DIR = RAW_DATA_DIR / "app_logs"

CACHE_DIR = PROJECT_ROOT / "cache" / "jikan"

JIKAN_BASE_URL = "https://api.jikan.moe/v4"
JIKAN_REQUESTS_PER_MINUTE = 55
JIKAN_REQUESTS_PER_SECOND = 2.5
JIKAN_REQUEST_DELAY = 1 / JIKAN_REQUESTS_PER_SECOND
JIKAN_CACHE_EXPIRATION_DAYS = 7

RATING_THRESHOLD = 7
MIN_LIKES_PER_USER = 50
MIN_LIKES_PER_ANIME = 50
HOLDOUT_RATIO = 0.1

KAGGLE_ANIME_FILE = "Anime.csv"
KAGGLE_REVIEWS_FILE = "User-AnimeReview.csv"
UNIFIED_ANIME_FILE = "anime_catalog.parquet"
UNIFIED_INTERACTIONS_FILE = "interactions.parquet"
TRAINING_DATA_FILE = "train.parquet"
TEST_DATA_FILE = "test.npy"
METADATA_FILE = "metadata.json"
ANIME_ID_MAP_FILE = "anime_id_map.json"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

KAGGLE_DATASET_LOCATIONS = [
    KAGGLE_DATA_DIR,
    DATA_DIR / "datasets" / "bsurya27" / "myanimelists-anime-and-user-anime-interactions" / "versions" / "1" / "MyAnimeList-dataset",
    Path.home() / ".cache" / "kagglehub" / "datasets" / "bsurya27" / "myanimelists-anime-and-user-anime-interactions" / "versions" / "1",
]

def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINING_DATA_DIR,
        KAGGLE_DATA_DIR, JIKAN_DATA_DIR, APP_LOGS_DIR, CACHE_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def find_kaggle_data_dir() -> Path | None:
    """Find the directory containing Kaggle CSV files."""
    for location in KAGGLE_DATASET_LOCATIONS:
        anime_path = location / KAGGLE_ANIME_FILE
        reviews_path = location / KAGGLE_REVIEWS_FILE
        if anime_path.exists() and reviews_path.exists():
            return location
    return None

def get_kaggle_anime_path() -> Path:
    """Get path to Kaggle anime CSV file."""
    kaggle_dir = find_kaggle_data_dir()
    if kaggle_dir is None:
        raise FileNotFoundError(
            f"Kaggle data not found. Place Anime.csv and User-AnimeReview.csv in: {KAGGLE_DATA_DIR}"
        )
    return kaggle_dir / KAGGLE_ANIME_FILE

def get_kaggle_reviews_path() -> Path:
    """Get path to Kaggle reviews CSV file."""
    kaggle_dir = find_kaggle_data_dir()
    if kaggle_dir is None:
        raise FileNotFoundError(
            f"Kaggle data not found. Place Anime.csv and User-AnimeReview.csv in: {KAGGLE_DATA_DIR}"
        )
    return kaggle_dir / KAGGLE_REVIEWS_FILE

def get_processed_anime_path() -> Path:
    """Get path to processed anime catalog."""
    return PROCESSED_DATA_DIR / UNIFIED_ANIME_FILE

def get_processed_interactions_path() -> Path:
    """Get path to processed interactions file."""
    return PROCESSED_DATA_DIR / UNIFIED_INTERACTIONS_FILE

def get_training_data_path() -> Path:
    """Get path to training data file."""
    return TRAINING_DATA_DIR / TRAINING_DATA_FILE

def get_test_data_path() -> Path:
    """Get path to test data file."""
    return TRAINING_DATA_DIR / TEST_DATA_FILE
