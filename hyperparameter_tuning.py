import csv
import itertools
import os
import random

import numpy as np
import torch
import yaml

from src.data_loader import load_anime_dataset
from src.model import RBM
from src.train import train_rbm
from src.utils import preprocess_data, make_train_test_split

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
path_config = config["paths"]

param_grid = {
    "n_hidden": [512, 1024],
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32, 64],
}

# Load data
ratings, anime = load_anime_dataset()
user_anime, _ = preprocess_data(ratings, min_likes_user=data_config["min_likes_user"], min_likes_anime=data_config["min_likes_anime"])
train_df, test_array = make_train_test_split(user_anime, holdout_ratio=data_config["holdout_ratio"])

train_tensor = torch.FloatTensor(train_df.values)
test_tensor = torch.FloatTensor(test_array)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

best_map = -1
best_params = None

for combo in itertools.product(*param_grid.values()):
    n_hidden, learning_rate, batch_size = combo
    print(f"\nTesting: n_hidden={n_hidden}, lr={learning_rate}, batch_size={batch_size}")

    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)
    rbm, _, precisions, maps, ndcgs = train_rbm(
        rbm,
        train_tensor.to(device),
        test_tensor.to(device),
        epochs=30,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=10,
        device=device,
    )

    final_precision = precisions[-1]
    final_ndcg = ndcgs[-1]
    final_map = maps[-1]

    if final_map > best_map:
        best_map = final_map
        best_params = {
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }

    write_header = not os.path.exists("out/tuning_results.csv")
    with open("out/tuning_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "n_hidden", "learning_rate", "batch_size",
                "precision_at_10", "ndcg_at_10", "map_at_10"
            ])
        writer.writerow([
            n_hidden, learning_rate, batch_size,
            final_precision, final_ndcg, final_map
        ])

print("\nBest config:", best_params)
print("Best MAP@10:", best_map)