from __future__ import annotations
import pandas as pd
from typing import Tuple
from api.config import RATING_THRESHOLD

def filter_hentai(ratings: pd.DataFrame, anime: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = ~anime.apply(lambda row: row.astype(str).str.contains('Hentai', case=False, na=False)).any(axis=1)
    anime_clean = anime[mask]
    ratings_clean = ratings[ratings['anime_id'].isin(anime_clean['anime_id'])]
    return ratings_clean, anime_clean


def preprocess_data(ratings_df: pd.DataFrame, min_likes_user: int, min_likes_anime: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = ratings_df.copy()
    df = df[df['status'] == 'Completed']
    df['liked'] = (df['score'] >= RATING_THRESHOLD).astype(int)
    anime_likes = df.groupby('anime_id')['liked'].sum()
    active_anime = anime_likes[anime_likes >= min_likes_anime].index
    df = df[df['anime_id'].isin(active_anime)]
    user_likes = df.groupby('user_id')['liked'].sum()
    active_users = user_likes[user_likes >= min_likes_user].index
    df = df[df['user_id'].isin(active_users)]

    user_anime = df.pivot_table(
        index='user_id',
        columns='anime_id',
        values='liked',
        fill_value=0
    )

    return user_anime, df
