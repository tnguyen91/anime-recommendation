import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from constants import (
    DEFAULT_SEED, DEFAULT_N_HIDDEN, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_K, DEFAULT_TOP_N,
    CONFIG_FILE
)
from src.data_loader import load_anime_dataset
from src.model import RBM
from src.train import train_rbm
from src.utils import preprocess_data, make_train_test_split, generate_recommendations_csv, plot_training_metrics, interactive_recommender

try:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    data_config = config["data"]
    path_config = config["paths"]
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_FILE}' not found.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML configuration: {e}")
    exit(1)
except KeyError as e:
    print(f"Error: Missing required configuration section: {e}")
    exit(1)

np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

def load_and_preprocess_data(data_config):
    """
    Load datasets and preprocess for collaborative filtering.
    
    Loads anime ratings and metadata from Kaggle datasets, filters out
    adult content, and preprocesses into binary user-item interactions
    with minimum activity thresholds.

    Returns:
        tuple: (ratings, anime, user_anime)
    """
    ratings, anime = load_anime_dataset()

    if ratings.empty or anime.empty:
        raise ValueError("Dataset is empty after loading")

    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=data_config["min_likes_user"],
        min_likes_anime=data_config["min_likes_anime"]
    )
    
    if user_anime.empty:
        raise ValueError("No data remaining after preprocessing filters")
    
    print("user_anime shape:", user_anime.shape)
    print("ratings shape:", ratings.shape)
    return ratings, anime, user_anime

def prepare_train_test_data(user_anime, data_config):
    try:
        print("Creating train-test split...")
        train, test = make_train_test_split(user_anime, holdout_ratio=data_config["holdout_ratio"])
        held_out_counts = test.sum(axis=1)
        print("Test split stats:\n", pd.Series(held_out_counts).describe())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if train.empty or test.size == 0:
            raise ValueError("Train-test split resulted in empty data")

        train_tensor = torch.FloatTensor(train.values).to(device)
        test_tensor = torch.FloatTensor(test).to(device)
        
        return train, test, train_tensor, test_tensor, device
        
    except Exception as e:
        print(f"Error preparing train-test data: {e}")
        raise

def train_model_workflow(rbm, train_tensor, test_tensor, train, test, user_anime, anime, device, **kwargs):
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size'] 
    learning_rate = kwargs['learning_rate']
    k = kwargs['k']
    
    print(f"Training hyperparameter: \nepochs-{epochs}    batch_size-{batch_size}    learning_rate-{learning_rate}   n_hidden-{kwargs['n_hidden']}")
    
    rbm, losses, precs, maps, ndcgs = train_rbm(
        rbm, train_tensor, test_tensor,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        k=k,
        device=device
    )
    
    plot_training_metrics(losses, precs, maps, ndcgs, k)
    generate_recommendations_csv(rbm, train, test, user_anime, anime, device=device)
    return rbm

def load_pretrained_model(rbm, model_path, device):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        rbm = torch.quantization.quantize_dynamic(rbm, {torch.nn.Linear}, dtype=torch.qint8)
        rbm.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained RBM from {model_path}")
        return rbm
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first or specify the correct path.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main(train_model=True,
         run_cli=True,
         n_hidden=DEFAULT_N_HIDDEN,
         epochs=DEFAULT_EPOCHS,
         batch_size=DEFAULT_BATCH_SIZE,
         learning_rate=DEFAULT_LEARNING_RATE,
         k=DEFAULT_K,
         model_path='out/rbm_best_model.pth'):
    
    ratings, anime, user_anime = load_and_preprocess_data(data_config)
    
    train, test, train_tensor, test_tensor, device = prepare_train_test_data(user_anime, data_config)
    
    rbm = RBM(n_visible=train_tensor.shape[1], n_hidden=n_hidden).to(device)

    if train_model:
        rbm = train_model_workflow(
            rbm, train_tensor, test_tensor, train, test, user_anime, anime, device,
            n_hidden=n_hidden, epochs=epochs, batch_size=batch_size, 
            learning_rate=learning_rate, k=k
        )
    else:
        rbm = load_pretrained_model(rbm, model_path, device)
        if rbm is None:
            return
    
    if run_cli:
        interactive_recommender(user_anime, anime, rbm, device, top_n=DEFAULT_TOP_N)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime Recommendation System (RBM)")
    parser.add_argument('--train', action='store_true', help="Train the RBM model")
    parser.add_argument('--no-cli', action='store_true', help="Do not launch the interactive CLI recommender")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help="Training learning rate")
    parser.add_argument('--n-hidden', type=int, default=DEFAULT_N_HIDDEN, help="Number of hidden units in RBM")
    parser.add_argument('--k', type=int, default=DEFAULT_K, help="Top-K for evaluation/metrics")
    parser.add_argument('--model-path', type=str, default='out/rbm_best_model.pth', help="Path to save/load RBM model")
    args = parser.parse_args()

    main(
        train_model=args.train,
        run_cli=not args.no_cli,
        n_hidden=args.n_hidden if args.n_hidden != DEFAULT_N_HIDDEN else model_config["n_hidden"],
        epochs=args.epochs if args.epochs != DEFAULT_EPOCHS else model_config["epochs"],
        batch_size=args.batch_size if args.batch_size != DEFAULT_BATCH_SIZE else model_config["batch_size"],
        learning_rate=args.learning_rate if args.learning_rate != DEFAULT_LEARNING_RATE else model_config["learning_rate"],
        k=args.k if args.k != DEFAULT_K else model_config["k"],
        model_path=args.model_path if args.model_path != 'out/rbm_best_model.pth' else path_config["model_path"]
    )
