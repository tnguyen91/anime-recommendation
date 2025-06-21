import numpy as np
import pandas as pd
import torch

def filter_hentai(ratings, anime):
    mask = ~anime.apply(lambda row: row.astype(str).str.contains('hentai', case=False, na=False)).any(axis=1)
    anime_clean = anime[mask]
    ratings_clean = ratings[ratings['anime_id'].isin(anime_clean['MAL_ID'])]
    return ratings_clean, anime_clean

def preprocess_data(ratings_df, min_likes_user=700, min_likes_anime=100):
    ratings_df = ratings_df.copy()
    ratings_df['liked'] = (ratings_df['rating'] >= 7).astype(int)

    anime_likes = ratings_df.groupby('anime_id')['liked'].sum()
    active_anime = anime_likes[anime_likes >= min_likes_anime].index
    ratings_df = ratings_df[ratings_df['anime_id'].isin(active_anime)]

    user_likes = ratings_df.groupby('user_id')['liked'].sum()
    active_users = user_likes[user_likes >= min_likes_user].index
    ratings_df = ratings_df[ratings_df['user_id'].isin(active_users)]

    user_anime = ratings_df.pivot_table(
        index='user_id',
        columns='anime_id',
        values='liked',
        fill_value=0
    )

    return user_anime, ratings_df

def make_train_test_split(data, holdout_ratio=0.1, seed=1234):
    np.random.seed(seed)
    train = data.copy()
    test = np.zeros(data.shape)

    for user in range(data.shape[0]):
        liked_idx = np.where(data.iloc[user] == 1)[0]
        test_size = int(np.floor(len(liked_idx) * holdout_ratio))
        if test_size > 0:
            test_idx = np.random.choice(liked_idx, size=test_size, replace=False)
            train.iloc[user, test_idx] = 0
            test[user, test_idx] = 1
    return train, test

def get_recommendations(input_vector, rbm, anime_ids, anime_df, top_n=10, device='cpu'):
    rbm.eval()

    if isinstance(input_vector, (list, np.ndarray)):
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

    recommended = anime_df[anime_df['MAL_ID'].isin(top_anime_ids)].copy()
    recommended = recommended.set_index('MAL_ID').loc[top_anime_ids]
    recommended['score'] = scores[top_indices]

    return recommended[['Name', 'score']]
