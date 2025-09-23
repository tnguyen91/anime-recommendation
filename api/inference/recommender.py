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
    device: str = 'cpu',
) -> pd.DataFrame:
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

    top_indices = scores.argsort()[::-1][:top_n]
    top_anime_ids = [anime_ids[i] for i in top_indices]

    recommended = anime_df[anime_df['anime_id'].isin(top_anime_ids)].copy()
    recommended = recommended.set_index('anime_id').loc[top_anime_ids]
    recommended['score'] = scores[top_indices]
    recommended = recommended.reset_index()

    return recommended[['anime_id', 'name', 'score']]
