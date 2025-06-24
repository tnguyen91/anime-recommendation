from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split, get_recommendations, generate_recommendations_csv
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
    user_anime, rating = preprocess_data(ratings)
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
    plt.savefig("training_metrics.png")

    generate_recommendations_csv(rbm, train, test, user_anime, anime, device=device, top_n=10, filename="recommendations.csv")

if __name__ == "__main__":
    main()
