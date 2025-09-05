import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from typing import Tuple

from constants import (
    DEFAULT_SEED, RATING_THRESHOLD, DEFAULT_TOP_N,
    DEFAULT_FIGURE_SIZE
)

def filter_hentai(ratings: pd.DataFrame, anime: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = ~anime.apply(lambda row: row.astype(str).str.contains('Hentai', case=False, na=False)).any(axis=1)
    anime_clean = anime[mask]
    ratings_clean = ratings[ratings['anime_id'].isin(anime_clean['anime_id'])]
    return ratings_clean, anime_clean

def preprocess_data(ratings_df, min_likes_user=100, min_likes_anime=50):
    ratings_df = ratings_df.copy()
    ratings_df = ratings_df[ratings_df['status'] == 'Completed']
    ratings_df['liked'] = (ratings_df['score'] >= RATING_THRESHOLD).astype(int)

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

def make_train_test_split(data, holdout_ratio=0.1, seed=DEFAULT_SEED):
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

def get_recommendations(input_vector, rbm, anime_ids, anime_df, top_n=DEFAULT_TOP_N, device='cpu'):
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

    recommended = anime_df[anime_df['anime_id'].isin(top_anime_ids)].copy()
    recommended = recommended.set_index('anime_id').loc[top_anime_ids]
    recommended['score'] = scores[top_indices]
    recommended = recommended.reset_index()  # bring anime_id back as column

    return recommended[['anime_id', 'name', 'score']]
  

def generate_batch_recommendations(rbm, input_tensor, anime_ids, device, top_n):
    rbm.eval()
    with torch.no_grad():
        p_h, _ = rbm.sample_h(input_tensor)
        p_v, _ = rbm.sample_v(p_h)
        p_v[input_tensor == 1] = -1e6  # mask already-liked items
    
    scores = p_v.cpu().numpy()
    return scores

def process_user_recommendations(user_id, user_scores, anime_ids, anime_df, test_vector, top_n):
    top_indices = user_scores.argsort()[::-1][:top_n]
    top_anime_ids = [anime_ids[j] for j in top_indices]
    
    held_out_indices = np.where(test_vector.astype(int) == 1)[0]
    held_out_ids = [anime_ids[j] for j in held_out_indices]
    
    user_recommendations = []
    for j, anime_id in zip(top_indices, top_anime_ids):
        anime_name = 'Unknown'
        if anime_id in anime_df['anime_id'].values:
            anime_name = anime_df.loc[anime_df['anime_id'] == anime_id, 'name'].values[0]
        
        user_recommendations.append({
            'user_id': user_id,
            'anime_id': anime_id,
            'anime_name': anime_name,
            'predicted_score': user_scores[j],
            'is_held_out': anime_id in held_out_ids
        })
    
    return user_recommendations

def generate_recommendations_csv(rbm, train, test, user_anime, anime,
                                 device='cpu', top_n=DEFAULT_TOP_N, filename="out/recommendations.csv"):
    user_ids = list(user_anime.index)
    anime_ids = list(user_anime.columns)
    input_tensor = torch.FloatTensor(train.values).to(device)

    scores = generate_batch_recommendations(rbm, input_tensor, anime_ids, device, top_n)
    
    recommendation_rows = []
    for i, user_id in enumerate(user_ids):
        user_recs = process_user_recommendations(
            user_id, scores[i], anime_ids, anime, test[i], top_n
        )
        recommendation_rows.extend(user_recs)

    recommendation_df = pd.DataFrame(recommendation_rows)
    recommendation_df.to_csv(filename, index=False)
    print(f"Recommendations saved to {filename}")
    return recommendation_df

def search_anime(anime_df, query):
    """
    Search for anime by name across multiple title fields.
    
    Performs case-insensitive substring search across name, English title,
    and Japanese title fields to find matching anime entries.
    
    Args:
        anime_df (pd.DataFrame): Anime metadata dataframe
        query (str): Search query string
        
    Returns:
        pd.DataFrame: Matching anime with ID and title columns
    """
    name_cols = ["name", "title_english", "title_japanese"]
    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)
    matches = anime_df[mask]
    return matches[["anime_id"] + name_cols]

def make_input_vector(liked_anime_ids, anime_ids):
    """
    Create binary input vector from user's liked anime IDs.
    
    Converts a list of liked anime IDs into a binary vector compatible
    with the RBM model, where 1 indicates the user liked that anime.
    
    Args:
        liked_anime_ids (list): List of anime IDs the user has liked
        anime_ids (list): Complete list of anime IDs in the model
        
    Returns:
        np.ndarray: Binary vector of length len(anime_ids)
    """
    return [1 if anime_id in liked_anime_ids else 0 for anime_id in anime_ids]

def plot_training_metrics(losses, precs, maps, ndcgs, K):
    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.plot(losses, label="Loss")
    plt.plot(precs, label=f"Precision@{K}")
    plt.plot(maps, label=f"MAP@{K}")
    plt.plot(ndcgs, label=f"NDCG@{K}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("RBM Training Metrics")
    plt.legend()
    plt.grid(True)
    rbm_dir = os.path.dirname(os.path.dirname(__file__)) 
    output_path = os.path.join(rbm_dir, "out/training_metrics.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

def collect_user_preferences(anime_df):
    liked_anime_ids = []
    
    print("\n=== Anime Recommendation CLI ===")
    print("Search and select anime you like (press Enter without typing to finish):")
    
    while True:
        query = input("\nSearch anime: ").strip()
        if not query:
            break
            
        results = search_anime(anime_df, query)
        if results.empty:
            print("No matches found. Try another keyword.")
            continue
            
        print(results)
        chosen = input("Type the anime_id (comma separated) of anime you like: ").strip()
        if chosen:
            new_ids = [int(id_) for id_ in chosen.split(",") if id_.isdigit()]
            liked_anime_ids.extend(new_ids)
    
    return list(set(liked_anime_ids))

def interactive_recommender(user_anime, anime, rbm, device, top_n=DEFAULT_TOP_N):
    anime_ids = list(user_anime.columns)
    
    liked_anime_ids = collect_user_preferences(anime)
    
    if not liked_anime_ids:
        print("No anime selected. Exiting recommender.")
        return
    
    input_vector = make_input_vector(liked_anime_ids, anime_ids)
    input_vector_tensor = torch.FloatTensor(input_vector).to(device)
    recommendations = get_recommendations(input_vector_tensor, rbm, anime_ids, anime, top_n=top_n, device=device)
    
    print("\n=== Top Recommendations for You ===")
    print(recommendations)
    return recommendations