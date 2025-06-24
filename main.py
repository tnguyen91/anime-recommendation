from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split, get_recommendations
from src.model import RBM
from src.train import train_rbm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

N_HIDDEN = 512
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
K = 10

def main():
    print("Loading data...")
    ratings, anime = load_anime_dataset()
    user_anime, _ = preprocess_data(ratings)
    print(user_anime.shape)
    print(ratings.shape)

    print("Creating train-test split...")
    train, test = make_train_test_split(user_anime, holdout_ratio=0.1)
    held_out_counts = test.sum(axis=1)
    print(pd.Series(held_out_counts).describe())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training...(using {device})")

    train_tensor = torch.FloatTensor(train.values).to(device)
    test_tensor = torch.FloatTensor(test).to(device)

    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=512).to(device)
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


    user_ids = list(user_anime.index)
    input_tensor = torch.FloatTensor(train.values).to(device)

    with torch.no_grad():
        p_h, _ = rbm.sample_h(input_tensor)
        p_v, _ = rbm.sample_v(p_h)
        p_v[input_tensor == 1] = -1e6  # mask already-liked

    scores = p_v.cpu().numpy()
    anime_ids = list(user_anime.columns)
    recommendation_rows = []

    for i, user_id in enumerate(user_ids):
        user_scores = scores[i]
        top_indices = user_scores.argsort()[::-1][:10]
        top_anime_ids = [anime_ids[j] for j in top_indices]

        held_out_vector = test[i].astype(int)
        held_out_indices = np.where(held_out_vector == 1)[0]
        held_out_ids = [anime_ids[j] for j in held_out_indices]

        for j, anime_id in zip(top_indices, top_anime_ids):
            recommendation_rows.append({
                'user_id': user_id,
                'anime_id': anime_id,
                'anime_name': anime.loc[anime['MAL_ID'] == anime_id, 'Name'].values[0] if anime_id in anime['MAL_ID'].values else 'Unknown',
                'predicted_score': user_scores[j],
                'is_held_out': anime_id in held_out_ids
            })
    recommendation_df = pd.DataFrame(recommendation_rows)
    recommendation_df.to_csv("recommendations.csv", index=False)
    print("Recommendations saved to recommendations.csv")

if __name__ == "__main__":
    main()
