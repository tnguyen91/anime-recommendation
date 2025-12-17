"""Grid search hyperparameter tuning with resume capability."""
import csv
import itertools
import os
import random as _random
import numpy as np
import torch
import yaml
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rbm.constants import HYPERPARAMETER_EPOCHS, HYPERPARAMETER_GRID, SEED, CONFIG_FILE
DEFAULT_K = 10
from rbm.src.data_loader import load_anime_dataset
from rbm.src.model import RBM
from rbm.src.train import train_rbm
from rbm.src.utils import preprocess_data, make_train_test_split

_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
param_grid = HYPERPARAMETER_GRID
target_n_hidden = os.environ.get("TARGET_N_HIDDEN")
if target_n_hidden:
    try:
        requested = [int(x.strip()) for x in target_n_hidden.split(",") if x.strip()]
        param_grid = dict(param_grid)
        param_grid["n_hidden"] = [v for v in param_grid.get("n_hidden", []) if v in requested]
    except Exception:
        print("Warning: failed to parse TARGET_N_HIDDEN; ignoring")

ratings, anime = load_anime_dataset()
_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
try:
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass
user_anime, _ = preprocess_data(
    ratings,
    min_likes_user=data_config["min_likes_user"],
    min_likes_anime=data_config["min_likes_anime"]
)
train_df, test_array = make_train_test_split(user_anime, holdout_ratio=data_config["holdout_ratio"], seed=SEED)
train_tensor = torch.FloatTensor(train_df.values)
test_tensor = torch.FloatTensor(test_array)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

completed_configs = set()
results_file = "out/tuning_results.csv"
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["n_hidden"]), float(row["learning_rate"]), int(row["batch_size"]))
            completed_configs.add(key)
    print(f"Resuming: {len(completed_configs)} configs already completed, skipping them.")

best_map = -1
best_params = None

all_combos = list(itertools.product(param_grid["n_hidden"], param_grid["learning_rate"], param_grid["batch_size"]))
total_configs = len(all_combos)
remaining_configs = [(i, c) for i, c in enumerate(all_combos) if c not in completed_configs]

print(f"Total configs: {total_configs}, Remaining: {len(remaining_configs)}")

for idx, combo in remaining_configs:
    n_hidden, learning_rate, batch_size = combo
    print(f"\n[{idx + 1}/{total_configs}] Testing: n_hidden={n_hidden}, lr={learning_rate}, batch_size={batch_size}")
    _random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)
    rbm, losses, precs, maps, ndcgs = train_rbm(
        rbm, train_tensor, test_tensor,
        epochs=HYPERPARAMETER_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=DEFAULT_K,
        device=device
    )
    final_precision = precs[-1]
    final_ndcg = ndcgs[-1]
    final_map = maps[-1]
    if final_map > best_map:
        best_map = final_map
        best_params = {"n_hidden": n_hidden, "learning_rate": learning_rate, "batch_size": batch_size}
    write_header = not os.path.exists(results_file)
    os.makedirs("out", exist_ok=True)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["n_hidden", "learning_rate", "batch_size", "precision_at_10", "ndcg_at_10", "map_at_10"])
        writer.writerow([n_hidden, learning_rate, batch_size, final_precision, final_ndcg, final_map])

print("\nBest config:", best_params)
print("Best MAP@10:", best_map)
