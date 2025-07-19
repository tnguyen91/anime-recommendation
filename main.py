from src.data_loader import load_anime_dataset
from src.utils import preprocess_data, make_train_test_split, generate_recommendations_csv, plot_training_metrics, interactive_recommender
from src.model_factory import get_model
import torch
import numpy as np
import pandas as pd
import os
import argparse
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_config = config["model"]
data_config = config["data"]
path_config = config["paths"]
SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)

def main(model_type='rbm',
         train_model=True,
         run_cli=True,
         n_hidden=512,
         epochs=20,
         batch_size=32,
         learning_rate=0.001,
         k=10,
         model_path='out/rbm_best_model.pth'):
    print("Loading data...")
    ratings, anime = load_anime_dataset()
    user_anime, _ = preprocess_data(ratings, min_likes_user=data_config["min_likes_user"], min_likes_anime=data_config["min_likes_anime"])
    print("user_anime shape:", user_anime.shape)
    print("ratings shape:", ratings.shape)

    print("Creating train-test split...")
    train, test = make_train_test_split(user_anime, holdout_ratio=data_config["holdout_ratio"])
    held_out_counts = test.sum(axis=1)
    print("Test split stats:\n", pd.Series(held_out_counts).describe())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tensor = torch.FloatTensor(train.values).to(device)
    test_tensor = torch.FloatTensor(test).to(device)

    model_params = {
        "n_visible": train_tensor.shape[1],
        "n_hidden": n_hidden
    }

    model = get_model(model_type, **model_params).to(device)
    
    if train_model:
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
        print(f"Training hyperparameter: \nepochs-{epochs}    batch_size-{batch_size}    learning_rate-{learning_rate}   n_hidden-{n_hidden}")
        results =  model.fit(
            train_tensor,
            test_tensor,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            k=k,
            device=device
        ) 
        plot_training_metrics(results['losses'], results['precs'], results['maps'], results['ndcgs'], k)
        generate_recommendations_csv(model, train, test, user_anime, anime, device=device)
        model.save(model_path)
    else:
        model.load(model_path, device=device)
    if run_cli:
        interactive_recommender(user_anime, anime, model, device, top_n=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime Recommendation System (RBM)")
    parser.add_argument('--model', type=str, default='rbm', help="Model type to use (e.g., rbm)");
    parser.add_argument('--train', action='store_true', help="Train the RBM model")
    parser.add_argument('--no-cli', action='store_true', help="Do not launch the interactive CLI recommender")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Training batch size")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="Training learning rate")
    parser.add_argument('--n-hidden', type=int, default=512, help="Number of hidden units in RBM")
    parser.add_argument('--k', type=int, default=10, help="Top-K for evaluation/metrics")
    parser.add_argument('--model-path', type=str, default='out/rbm_best_model.pth', help="Path to save/load RBM model")
    args = parser.parse_args()

    main(
        model_type=args.model if args.model != 'rbm' else model_config.get("type", "rbm"),
        train_model=args.train,
        run_cli=not args.no_cli,
        n_hidden=args.n_hidden if args.n_hidden != 512 else model_config["n_hidden"],
        epochs=args.epochs if args.epochs != 20 else model_config["epochs"],
        batch_size=args.batch_size if args.batch_size != 32 else model_config["batch_size"],
        learning_rate=args.learning_rate if args.learning_rate != 0.001 else model_config["learning_rate"],
        k=args.k if args.k != 10 else model_config["k"],
        model_path=args.model_path if args.model_path != 'out/rbm_best_model.pth' else path_config["model_path"]
    )
