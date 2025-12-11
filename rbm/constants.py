import os
from typing import Dict, List

HTTP_BAD_REQUEST = 400

RATING_THRESHOLD = 7

CLAMP_MIN = 0.0
CLAMP_MAX = 1.0
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 3

KAGGLE_DATASET = "bsurya27/myanimelists-anime-and-user-anime-interactions"

_RBM_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_RBM_DIR, '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "out")

DATA_VERSIONS_PATH = os.path.join(
    DATA_DIR,
    'datasets',
    'bsurya27',
    'myanimelists-anime-and-user-anime-interactions',
    'versions',
    '*/',
    'MyAnimeList-dataset'
)
API_DELAY_SECONDS = 0.5

CONFIG_FILE = "config.yaml"
ANIME_METADATA_FILE = os.path.join(DATA_DIR, "anime_metadata.json") 

DEFAULT_FIGURE_SIZE = (10, 6)
SEED = 1234
DEFAULT_TOP_N = 10

HYPERPARAMETER_GRID: Dict[str, List] = {
    "n_hidden": [256, 512, 1024],
    "learning_rate": [0.005, 0.001],
    "batch_size": [16, 32, 64],
}
HYPERPARAMETER_EPOCHS = 50
