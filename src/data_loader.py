import glob
import os

import kagglehub
import numpy as np
import pandas as pd

from src.utils import filter_hentai

os.environ["KAGGLEHUB_CACHE"] = "./data"

def load_anime_dataset():
    path = kagglehub.dataset_download("bsurya27/myanimelists-anime-and-user-anime-interactions")
    versions_dir = "./data/datasets/bsurya27/myanimelists-anime-and-user-anime-interactions/versions/*/MyAnimeList-dataset/"
    latest_version = sorted(glob.glob(versions_dir))[-1].replace("\\", "/")

    rating_path = os.path.join(latest_version, "User-AnimeReview.csv")
    anime_path = os.path.join(latest_version, "Anime.csv")
    
    ratings = pd.read_csv(rating_path)
    anime = pd.read_csv(anime_path)

    return filter_hentai(ratings, anime)
