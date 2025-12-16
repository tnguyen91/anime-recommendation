"""
Data preprocessing utilities for the recommendation system.

Handles filtering and transforming raw anime/ratings data into a format
suitable for RBM training and inference. Key preprocessing steps:
- Remove adult content
- Filter inactive users (< threshold reviews)
- Filter unpopular anime (< threshold likes)
- Convert ratings to binary liked/not-liked
"""
from __future__ import annotations
import pandas as pd
from typing import Tuple
from api.config import RATING_THRESHOLD


def filter_hentai(ratings: pd.DataFrame, anime: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove adult content from datasets.

    Filters out any anime with 'Hentai' in any column and removes
    corresponding ratings.

    Args:
        ratings: User ratings DataFrame
        anime: Anime metadata DataFrame

    Returns:
        Tuple of (filtered_ratings, filtered_anime)
    """
    mask = ~anime.apply(lambda row: row.astype(str).str.contains('Hentai', case=False, na=False)).any(axis=1)
    anime_clean = anime[mask]
    ratings_clean = ratings[ratings['anime_id'].isin(anime_clean['anime_id'])]
    return ratings_clean, anime_clean


def filter_data(
    ratings_df: pd.DataFrame,
    min_likes_user: int,
    min_likes_anime: int,
) -> tuple[list[int], pd.DataFrame]:
    """
    Filter data to active users and popular anime for model training.

    Applies several filters to reduce sparsity and improve model quality:
    1. Keep only completed anime (watched to end)
    2. Convert scores to binary (liked if >= RATING_THRESHOLD)
    3. Keep anime with sufficient likes
    4. Keep users with sufficient likes

    Args:
        ratings_df: Raw ratings DataFrame
        min_likes_user: Minimum likes required per user
        min_likes_anime: Minimum likes required per anime

    Returns:
        Tuple of (active_anime_ids, filtered_ratings_df)
    """
    df = ratings_df.copy()
    df = df[df['status'] == 'Completed']
    df['liked'] = (df['score'] >= RATING_THRESHOLD).astype(int)
    anime_likes = df.groupby('anime_id')['liked'].sum()
    active_anime = anime_likes[anime_likes >= min_likes_anime].index
    df = df[df['anime_id'].isin(active_anime)]
    user_likes = df.groupby('user_id')['liked'].sum()
    active_users = user_likes[user_likes >= min_likes_user].index
    df = df[df['user_id'].isin(active_users)]

    return list(active_anime.astype(int)), df
