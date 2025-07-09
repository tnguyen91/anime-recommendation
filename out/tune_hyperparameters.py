import itertools
import torch
from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split
from src.model import RBM
from src.train import train_rbm
import csv
import os

param_grid = {
    "n_hidden": [128, 256, 512, 1024],
    "learning_rate": [0.01, 0.001, 0.0001],
    "batch_size": [16, 32, 64],
}

# Load data
ratings, anime = load_anime_dataset()
user_anime, _ = preprocess_data(ratings)
train_df, test_array = make_train_test_split(user_anime, holdout_ratio=0.1)

train_tensor = torch.FloatTensor(train_df.values)
test_tensor = torch.FloatTensor(test_array)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_map = -1
best_params = None

for combo in itertools.product(*param_grid.values()):
    n_hidden, learning_rate, batch_size = combo
    print(f"\nTesting: n_hidden={n_hidden}, lr={learning_rate}, batch_size={batch_size}")

    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)
    rbm, _, _, maps, _ = train_rbm(
        rbm,
        train_tensor.to(device),
        test_tensor.to(device),
        epochs=15,  # Use small number for tuning
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=10,
        device=device,
    )

    final_map = maps[-1]
    if final_map > best_map:
        best_map = final_map
        best_params = {
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }

    write_header = not os.path.exists("tuning_results.csv")
    with open("tuning_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["n_hidden", "learning_rate", "batch_size", "map_at_10"])
        writer.writerow([n_hidden, learning_rate, batch_size, final_map])

print("\nBest config:", best_params)
print("Best MAP@10:", best_map)

