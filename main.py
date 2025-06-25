from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split, get_recommendations, generate_recommendations_csv
from src.model import RBM
from src.train import train_rbm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===
SEED = 1234
N_HIDDEN = 512
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
K = 10

np.random.seed(SEED)
torch.manual_seed(SEED)

def search_anime(anime_df, query, limit=5):
    """Search anime by query in 'Name', 'English name', and 'Japanese name' columns."""
    name_cols = ["Name", "English name", "Japanese name"]
    mask = False
    for col in name_cols:
        mask = mask | anime_df[col].astype(str).str.contains(query, case=False, na=False)
    matches = anime_df[mask]
    return matches[["MAL_ID"] + name_cols].head(limit)

def make_input_vector(liked_anime_ids, anime_ids):
    """Create a binary vector for the RBM given user liked anime IDs."""
    return [1 if anime_id in liked_anime_ids else 0 for anime_id in anime_ids]

def interactive_recommender(user_anime, anime, rbm, device, top_n=10):
    """CLI for user-driven recommendations."""
    anime_ids = list(user_anime.columns)
    liked_anime_ids = []

    print("\n=== Anime Recommendation CLI ===")
    print("Search and select anime you like (press Enter without typing to finish):")
    while True:
        query = input("\nSearch anime: ").strip()
        if not query:
            break
        results = search_anime(anime, query)
        if results.empty:
            print("No matches found. Try another keyword.")
            continue
        print(results)
        chosen = input("Type the MAL_IDs (comma separated) of anime you like: ").strip()
        if chosen:
            liked_anime_ids.extend([int(id_) for id_ in chosen.split(",") if id_.isdigit()])

    liked_anime_ids = list(set(liked_anime_ids))
    if not liked_anime_ids:
        print("No anime selected. Exiting recommender.")
        return

    input_vector = make_input_vector(liked_anime_ids, anime_ids)
    input_vector_tensor = torch.FloatTensor(input_vector).to(device)
    recs = get_recommendations(input_vector_tensor, rbm, anime_ids, anime, top_n=top_n, device=device)
    print("\n=== Top Recommendations for You ===")
    print(recs)
    return recs

def plot_training_metrics(losses, precs, maps, ndcgs, K):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss")
    plt.plot(precs, label=f"Precision@{K}")
    plt.plot(maps, label=f"MAP@{K}")
    plt.plot(ndcgs, label=f"NDCG@{K}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("RBM Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_metrics.png")
    plt.show()

def main(train_model=True, run_cli=True):
    print("Loading data...")
    ratings, anime = load_anime_dataset()
    user_anime, _ = preprocess_data(ratings)
    print("user_anime shape:", user_anime.shape)
    print("ratings shape:", ratings.shape)

    print("Creating train-test split...")
    train, test = make_train_test_split(user_anime, holdout_ratio=0.1)
    held_out_counts = test.sum(axis=1)
    print("Test split stats:\n", pd.Series(held_out_counts).describe())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tensor = torch.FloatTensor(train.values).to(device)
    test_tensor = torch.FloatTensor(test).to(device)

    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=N_HIDDEN).to(device)

    if train_model:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        rbm, losses, precs, maps, ndcgs = train_rbm(
            rbm, train_tensor, test_tensor,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            k=K,
            device=device
        )
        plot_training_metrics(losses, precs, maps, ndcgs, K)
        generate_recommendations_csv(rbm, train, test, user_anime, anime, device=device)
    else:
        # LOAD TRAINED MODEL WEIGHTS
        model_path = "rbm_best_model.pth"
        if os.path.exists(model_path):
            rbm.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded trained RBM from {model_path}")
        else:
            print(f"Trained model '{model_path}' not found. Please train first or specify the correct path.")
            return
    if run_cli:
        interactive_recommender(user_anime, anime, rbm, device, top_n=10)

if __name__ == "__main__":
    main(train_model=False)
