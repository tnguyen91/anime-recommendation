"""Recommendation generation using the trained RBM model."""
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
    """Generate top-N anime recommendations from RBM reconstruction probabilities."""
    rbm.eval()

    if isinstance(input_vector, (list, tuple, np.ndarray)):
        input_vector = torch.FloatTensor(input_vector)
    input_vector = input_vector.unsqueeze(0).to(device)

    with torch.no_grad():
        p_h, _ = rbm.sample_h(input_vector)
        p_v, _ = rbm.sample_v(p_h)

    scores = p_v.squeeze().cpu().numpy()

    liked_indices = np.where(input_vector.cpu().numpy().flatten() == 1)[0]
    scores[liked_indices] = -1e6

    if exclude_ids:
        anime_id_to_idx = {aid: idx for idx, aid in enumerate(anime_ids)}
        for exclude_id in exclude_ids:
            if exclude_id in anime_id_to_idx:
                scores[anime_id_to_idx[exclude_id]] = -1e6

    top_indices = scores.argsort()[::-1][:top_n]
    top_anime_ids = [anime_ids[i] for i in top_indices]

    recommended = anime_df[anime_df['anime_id'].isin(top_anime_ids)].copy()

    existing_ids = [aid for aid in top_anime_ids if aid in recommended['anime_id'].values]
    if not existing_ids:
        return pd.DataFrame(columns=['anime_id', 'name', 'score'])

    recommended = recommended.set_index('anime_id').loc[existing_ids]
    id_to_score = {anime_ids[i]: scores[i] for i in top_indices}
    recommended['score'] = [id_to_score[aid] for aid in existing_ids]
    recommended = recommended.reset_index()

    cols = ['anime_id', 'name', 'score']
    if 'title_english' in recommended.columns:
        cols.append('title_english')
    if 'title_japanese' in recommended.columns:
        cols.append('title_japanese')

    return recommended[cols]
