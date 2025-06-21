from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split, get_recommendations
from src.model import RBM
from src.train import train_rbm
from src.evaluate import evaluate_at_k
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
    user_anime, rating = preprocess_data(ratings)
    print(user_anime.shape)
    print(ratings.shape)

    print("Creating train-test split...")
    train, test = make_train_test_split(user_anime, holdout_ratio=0.1)
    held_out_counts = test.sum(axis=1)
    print(pd.Series(held_out_counts).describe())

    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tensor = torch.FloatTensor(train.values)
    test_tensor = torch.FloatTensor(test)

    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=512)

    rbm, losses, precs, maps, ndcgs = train_rbm(
        rbm, train_tensor, test_tensor,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        k=10,
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
    plt.show()


    anime_ids = list(user_anime.columns)
    recommendation_rows = []

    for user_id in user_anime.index:
        input_vector = train.loc[user_id].values.astype(np.float32)

        user_index = list(user_anime.index).index(user_id)
        held_out_vector = test[user_index].astype(int)

        recs = get_recommendations(input_vector, rbm, anime_ids, anime_df=anime, top_n=10, device=device)
        recs = recs.reset_index()

        held_out_indices = np.where(held_out_vector == 1)[0]
        held_out_ids = [anime_ids[i] for i in held_out_indices]

        recs['is_held_out'] = recs['MAL_ID'].isin(held_out_ids)

        for _, row in recs.iterrows():
            recommendation_rows.append({
                'user_id': user_id,
                'anime_id': row['MAL_ID'],
                'anime_name': row['Name'],
                'predicted_score': row['score'],
                'is_held_out': row['is_held_out']
            })

    recommendation_df = pd.DataFrame(recommendation_rows)
    recommendation_df.to_csv("recommendations.csv", index=False)
    print("Recommendations saved to recommendations.csv")

if __name__ == "__main__":
    main()
