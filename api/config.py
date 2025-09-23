from __future__ import annotations

from pathlib import Path

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "out"
CACHE_DIR_DEFAULT = Path("/tmp/anime-cache")

HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500
HTTP_OK = 200

RATING_THRESHOLD = 7
DEFAULT_TOP_N = 10
MIN_LIKES_USER = 100
MIN_LIKES_ANIME = 50
N_HIDDEN = 1024

KAGGLE_DATASET = "bsurya27/myanimelists-anime-and-user-anime-interactions"
DATA_VERSIONS_PATH = str(
    DATA_DIR
    / "datasets"
    / "bsurya27"
    / "myanimelists-anime-and-user-anime-interactions"
    / "versions"
    / "*/"
    / "MyAnimeList-dataset"
)

LOCAL_MODEL_PATH = OUTPUT_DIR / "rbm_best_model.pth"
LOCAL_METADATA_PATH = DATA_DIR / "anime_metadata.json"
DEFAULT_MODEL_PATH = LOCAL_MODEL_PATH
DEFAULT_METADATA_PATH = LOCAL_METADATA_PATH
