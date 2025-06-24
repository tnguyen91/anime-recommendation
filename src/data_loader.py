import os
import glob
import pandas as pd
import numpy as np
import kagglehub
from src.utils import filter_hentai
os.environ["KAGGLEHUB_CACHE"] = "./data"

def load_anime_dataset():
    path = kagglehub.dataset_download("hernan4444/anime-recommendation-database-2020")
    versions_dir = "./data/datasets/hernan4444/anime-recommendation-database-2020/versions/*/"
    latest_version = sorted(glob.glob(versions_dir))[-1].replace("\\", "/")

    rating_path = os.path.join(latest_version, "rating_complete.csv")
    anime_path = os.path.join(latest_version, "anime.csv")
    
    ratings = pd.read_csv(rating_path)
    anime = pd.read_csv(anime_path)

    ratings['user_id'] = ratings['user_id'].astype(np.uint32)
    ratings['anime_id'] = ratings['anime_id'].astype(np.uint32)
    ratings['rating'] = ratings['rating'].astype(np.int8)

    return filter_hentai(ratings, anime)
