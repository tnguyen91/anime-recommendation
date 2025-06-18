import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import torch
from torch.utils.data import random_split

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_anime_data(n_users=5000, n_items=3000):
    os.environ["KAGGLEHUB_CACHE"] = "./data"
    path = kagglehub.dataset_download("dbdmobile/myanimelist-dataset")

    versions_dir = "../data/datasets/dbdmobile/myanimelist-dataset/versions/*/"
    version_folders = glob.glob(versions_dir)
    latest_version = sorted(version_folders)[-1].replace("\\", "/")

    rating_path = os.path.join(latest_version, "users-score-2023.csv")

    ratings = pd.read_csv(rating_path, chunksize=1500000)
    ratings = next(ratings)

    ratings['liked'] = (ratings['rating'] >= 7).astype(int)

    anime_counts = ratings['anime_id'].value_counts()
    popular_anime_ids = anime_counts[anime_counts >= 50].index
    ratings = ratings[ratings['anime_id'].isin(popular_anime_ids)]

    print("Filtered ratings shape:", ratings.shape)
    print(ratings.head())

    top_users = ratings['user_id'].value_counts().head(n_users).index

    top_anime = ratings['anime_id'].value_counts().head(n_items).index

    filtered_ratings = ratings[
        ratings['user_id'].isin(top_users) &
        ratings['anime_id'].isin(top_anime)
    ]

    user_anime = filtered_ratings.pivot_table(
        index='user_id',
        columns='anime_id',
        values='liked',
        fill_value=0
    )

    print(user_anime)

    return user_anime, ratings
