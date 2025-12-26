"""RBM training and interactive recommendation CLI."""
import os
import argparse
import random as _random
import numpy as np
import torch
import yaml
import mlflow

try:
    from rbm.constants import SEED, DEFAULT_TOP_N, CONFIG_FILE, OUTPUT_DIR
    from rbm.src.data_loader import load_anime_dataset
    from rbm.src.model import RBM
    from rbm.src.train import train_rbm
    from rbm.src.utils import (
        preprocess_data, make_train_test_split, plot_training_metrics,
        interactive_recommender
    )
except Exception:
    from constants import SEED, DEFAULT_TOP_N, CONFIG_FILE, OUTPUT_DIR
    from src.data_loader import load_anime_dataset
    from src.model import RBM
    from src.train import train_rbm
    from src.utils import (
        preprocess_data, make_train_test_split, plot_training_metrics,
        interactive_recommender
    )

MLFLOW_EXPERIMENT_NAME = "anime-rbm-training"
_mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
MLFLOW_TRACKING_URI = f"file:///{_mlruns_path.replace(os.sep, '/')}"

def load_config():
    """Load YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

model_cfg = config['model']
data_cfg = config['data']
path_cfg = config['paths']

for k, v in list(path_cfg.items()):
    if isinstance(v, str) and not os.path.isabs(v):
        if v.startswith('out'):
            path_cfg[k] = os.path.normpath(os.path.join(OUTPUT_DIR, os.path.relpath(v, 'out')))
        else:
            path_cfg[k] = os.path.normpath(os.path.join(os.path.dirname(__file__), v))

def load_and_preprocess_data():
    """Load datasets and preprocess into user-anime matrix."""
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
        min_likes_user=data_cfg["min_likes_user"],
        min_likes_anime=data_cfg["min_likes_anime"]
    )
    return ratings, anime, user_anime


def prepare_train_test_data(user_anime):
    """Split data and convert to tensors for training."""
    train_df, test_arr = make_train_test_split(user_anime, holdout_ratio=data_cfg["holdout_ratio"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.FloatTensor(train_df.values).to(device)
    test_tensor = torch.FloatTensor(test_arr).to(device)
    return train_df, test_arr, train_tensor, test_tensor, device


def train_workflow(rbm, train_tensor, test_tensor, train_df, test_arr, user_anime, anime, device, **kwargs):
    """Execute training loop with MLflow tracking and save metrics plot."""
    print(f"Training RBM with n_hidden={kwargs['n_hidden']}, learning_rate={kwargs['learning_rate']}, "
          f"batch_size={kwargs['batch_size']}, epochs={kwargs['epochs']}, k={kwargs['k']} on {device}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"rbm_h{kwargs['n_hidden']}_lr{kwargs['learning_rate']}_bs{kwargs['batch_size']}"

    with mlflow.start_run(run_name=run_name):
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow run name: {run_name}")

        mlflow.log_params({
            "min_likes_user": data_cfg["min_likes_user"],
            "min_likes_anime": data_cfg["min_likes_anime"],
            "holdout_ratio": data_cfg["holdout_ratio"],
            "seed": SEED
        })

        rbm, losses, precs, maps, ndcgs = train_rbm(
            rbm, train_tensor, test_tensor,
            epochs=kwargs['epochs'],
            batch_size=kwargs['batch_size'],
            learning_rate=kwargs['learning_rate'],
            k=kwargs['k'],
            device=device,
            use_mlflow=True
        )

        plot_path = plot_training_metrics(losses, precs, maps, ndcgs, kwargs['k'])
        if plot_path and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path="plots")

        print("=" * 60)
        print("Training complete. View results with: mlflow ui")
        print("=" * 60)

    return rbm


def main(train_model=True, run_cli=True, n_hidden=None, epochs=None,
         batch_size=None, learning_rate=None, k=None):
    """Main entry point for training and/or interactive CLI."""
    n_hidden = n_hidden or model_cfg['n_hidden']
    epochs = epochs or model_cfg['epochs']
    batch_size = batch_size or model_cfg['batch_size']
    learning_rate = learning_rate or model_cfg['learning_rate']
    k = k or model_cfg['k']
    
    ratings, anime, user_anime = load_and_preprocess_data()
    train_df, test_arr, train_tensor, test_tensor, device = prepare_train_test_data(user_anime)
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

    if train_model:
        rbm = train_workflow(
            rbm, train_tensor, test_tensor, train_df, test_arr, user_anime, anime, device,
            n_hidden=n_hidden, epochs=epochs, batch_size=batch_size, 
            learning_rate=learning_rate, k=k
        )
    else:
        if os.path.exists(path_cfg['model_path']):
            print(f"Loading model from {path_cfg['model_path']}")
            rbm.load_state_dict(torch.load(path_cfg['model_path'], map_location=device))
        else:
            print(f"Model path {path_cfg['model_path']} does not exist; exiting.")
            return

    if run_cli:
        interactive_recommender(user_anime, anime, rbm, device, top_n=DEFAULT_TOP_N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBM Anime Recommender")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--no-cli', action='store_true')
    parser.add_argument('--epochs', type=int, default=model_cfg.get('epochs'))
    parser.add_argument('--batch-size', type=int, default=model_cfg.get('batch_size'))
    parser.add_argument('--learning-rate', type=float, default=model_cfg.get('learning_rate'))
    parser.add_argument('--n-hidden', type=int, default=model_cfg.get('n_hidden'))
    parser.add_argument('--k', type=int, default=model_cfg.get('k'))
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
