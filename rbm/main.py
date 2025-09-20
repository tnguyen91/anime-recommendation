import os
import argparse
import numpy as np
import torch
import yaml

from constants import (
    SEED, N_HIDDEN, EPOCHS, BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_K, DEFAULT_TOP_N, CONFIG_FILE, OUTPUT_DIR
)
from src.data_loader import load_anime_dataset
from src.model import RBM
from src.train import train_rbm
from src.utils import (
    preprocess_data, make_train_test_split, plot_training_metrics,
    interactive_recommender
)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

model_cfg = config['model']
data_cfg = config['data']
path_cfg = config['paths']

np.random.seed(SEED)
torch.manual_seed(SEED)


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


def main(train_model=True, run_cli=True, n_hidden=None, epochs=None,
         batch_size=None, learning_rate=None, k=None):
    n_hidden = n_hidden or model_cfg['n_hidden']
    epochs = epochs or model_cfg['epochs']
    batch_size = batch_size or model_cfg['batch_size']
    learning_rate = learning_rate or model_cfg['learning_rate']
    k = k or model_cfg['k']
    
    ratings, anime, user_anime = load_and_preprocess_data()
    train_df, test_arr, train_tensor, test_tensor, device = prepare_train_test_data(user_anime)
    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)

    if train_model:
        rbm = train_workflow(
            rbm, train_tensor, test_tensor, train_df, test_arr, user_anime, anime, device,
            n_hidden=n_hidden, epochs=epochs, batch_size=batch_size, 
            learning_rate=learning_rate, k=k
        )
    else:
        if os.path.exists(path_cfg['model_path']):
            rbm.load_state_dict(torch.load(path_cfg['model_path'], map_location=device))
        else:
            return

    if run_cli:
        interactive_recommender(user_anime, anime, rbm, device, top_n=DEFAULT_TOP_N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBM Anime Recommender")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-cli', action='store_true')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--n-hidden', type=int, default=N_HIDDEN)
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--model-path', type=str, default=os.path.join(OUTPUT_DIR, 'rbm_best_model.pth'))
    args = parser.parse_args()
    main(
        train_model=args.train,
        run_cli=not args.no_cli,
        n_hidden=args.n_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        k=args.k
    )
