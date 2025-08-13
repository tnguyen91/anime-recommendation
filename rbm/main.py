import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml

from constants import (
    DEFAULT_SEED, DEFAULT_N_HIDDEN, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_K, DEFAULT_TOP_N, CONFIG_FILE
)
from src.data_loader import load_anime_dataset
from src.model import RBM
from src.train import train_rbm
from src.utils import (
    preprocess_data, make_train_test_split, plot_training_metrics,
    interactive_recommender
)

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]
data_cfg = cfg["data"]
path_cfg = cfg["paths"]

np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)


def load_and_preprocess_data():
    ratings, anime = load_anime_dataset()
    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=data_cfg["min_likes_user"],
        min_likes_anime=data_cfg["min_likes_anime"]
    )
    return ratings, anime, user_anime


def prepare_train_test_data(user_anime):
    train_df, test_arr = make_train_test_split(user_anime, holdout_ratio=data_cfg["holdout_ratio"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.FloatTensor(train_df.values).to(device)
    test_tensor = torch.FloatTensor(test_arr).to(device)
    return train_df, test_arr, train_tensor, test_tensor, device


def train_workflow(rbm, train_tensor, test_tensor, train_df, test_arr, user_anime, anime, device, **kwargs):
    rbm, losses, precs, maps, ndcgs = train_rbm(
        rbm, train_tensor, test_tensor,
        epochs=kwargs['epochs'],
        batch_size=kwargs['batch_size'],
        learning_rate=kwargs['learning_rate'],
        k=kwargs['k'],
        device=device
    )
    plot_training_metrics(losses, precs, maps, ndcgs, kwargs['k'])
    return rbm


def main(train_model=True, run_cli=True, n_hidden=DEFAULT_N_HIDDEN, epochs=DEFAULT_EPOCHS,
         batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE, k=DEFAULT_K,
         model_path='out/rbm_best_model.pth'):
    ratings, anime, user_anime = load_and_preprocess_data()
    train_df, test_arr, train_tensor, test_tensor, device = prepare_train_test_data(user_anime)
    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)
    if train_model:
        rbm = train_workflow(
            rbm, train_tensor, test_tensor, train_df, test_arr, user_anime, anime, device,
            n_hidden=n_hidden, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, k=k
        )
    else:
        if os.path.exists(model_path):
            rbm.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Model path not found: {model_path}")
            return
    if run_cli:
        interactive_recommender(user_anime, anime, rbm, device, top_n=DEFAULT_TOP_N)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBM Anime Recommender")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-cli', action='store_true')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--n-hidden', type=int, default=DEFAULT_N_HIDDEN)
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--model-path', type=str, default='out/rbm_best_model.pth')
    args = parser.parse_args()
    main(
        train_model=args.train,
        run_cli=not args.no_cli,
        n_hidden=args.n_hidden if args.n_hidden != DEFAULT_N_HIDDEN else model_cfg['n_hidden'],
        epochs=args.epochs if args.epochs != DEFAULT_EPOCHS else model_cfg['epochs'],
        batch_size=args.batch_size if args.batch_size != DEFAULT_BATCH_SIZE else model_cfg['batch_size'],
        learning_rate=args.learning_rate if args.learning_rate != DEFAULT_LEARNING_RATE else model_cfg['learning_rate'],
        k=args.k if args.k != DEFAULT_K else model_cfg['k'],
    model_path=args.model_path if args.model_path != 'out/rbm_best_model.pth' else path_cfg['model_path']
    )
