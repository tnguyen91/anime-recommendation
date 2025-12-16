"""
Recommendation generation using the trained RBM model.

This module provides the core recommendation logic. Given a user's liked anime,
it uses the RBM to predict which other anime the user would likely enjoy.

The recommendation process:
1. Encode liked anime as binary input vector
2. Pass through RBM to get reconstruction probabilities
3. Mask already-liked and excluded anime
4. Return top-N highest scoring recommendations
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import torch

from api.config import DEFAULT_TOP_N


def get_recommendations(
    input_vector: Sequence[int] | torch.Tensor,
    rbm: torch.nn.Module,
    anime_ids: list[int],
    anime_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    exclude_ids: list[int] | None = None,
    device: str = 'cpu',
) -> pd.DataFrame:
    """
    Generate anime recommendations using the RBM model.

    Performs a forward pass through the RBM to compute reconstruction
    probabilities for all anime, then returns the highest-scoring ones
    that haven't been liked or excluded.

    Args:
        input_vector: Binary vector indicating liked anime (1) or not (0)
        rbm: Trained RBM model
        anime_ids: Ordered list of anime IDs matching input vector indices
        anime_df: DataFrame with anime metadata
        top_n: Number of recommendations to return
        exclude_ids: Anime IDs to exclude from results (e.g., previously shown)
        device: Torch device ('cpu' or 'cuda')

    Returns:
        DataFrame with columns: anime_id, name, score, and optional title columns
    """
    rbm.eval()

    # Convert input to tensor if needed
    if isinstance(input_vector, (list, tuple, np.ndarray)):
        input_vector = torch.FloatTensor(input_vector)
    input_vector = input_vector.unsqueeze(0).to(device)

    # Forward pass: visible -> hidden -> visible (reconstruction)
    with torch.no_grad():
        p_h, _ = rbm.sample_h(input_vector)
        p_v, _ = rbm.sample_v(p_h)

    scores = p_v.squeeze().cpu().numpy()

    # Mask already-liked anime (don't recommend what user already likes)
    liked_indices = np.where(input_vector.cpu().numpy().flatten() == 1)[0]
    scores[liked_indices] = -1e6

    # Mask excluded anime (previously shown recommendations)
    if exclude_ids:
        anime_id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}
        for exclude_id in exclude_ids:
            if exclude_id in anime_id_to_idx:
                scores[anime_id_to_idx[exclude_id]] = -1e6

    # Get top-N recommendations by score
    top_indices = scores.argsort()[::-1][:top_n]
    top_anime_ids = [anime_ids[i] for i in top_indices]

    # Build result DataFrame with metadata
    recommended = anime_df[anime_df['anime_id'].isin(top_anime_ids)].copy()
    recommended = recommended.set_index('anime_id').loc[top_anime_ids]
    recommended['score'] = scores[top_indices]
    recommended = recommended.reset_index()

    # Select output columns
    cols = ['anime_id', 'name', 'score']
    if 'title_english' in recommended.columns:
        cols.append('title_english')
    if 'title_japanese' in recommended.columns:
        cols.append('title_japanese')

    return recommended[cols]
