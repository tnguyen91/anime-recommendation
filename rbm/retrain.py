"""Automated model retraining with validation and comparison."""

import argparse
import json
import os
import sys
import shutil
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import mlflow

try:
    from rbm.constants import SEED, OUTPUT_DIR, CONFIG_FILE
    from rbm.src.data_loader import load_anime_dataset
    from rbm.src.model import RBM
    from rbm.src.train import train_rbm
    from rbm.src.evaluate import evaluate_at_k
    from rbm.src.utils import preprocess_data, make_train_test_split
    from rbm.main import load_config
except ImportError:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from constants import SEED, OUTPUT_DIR, CONFIG_FILE
    from src.data_loader import load_anime_dataset
    from src.model import RBM
    from src.train import train_rbm
    from src.evaluate import evaluate_at_k
    from src.utils import preprocess_data, make_train_test_split
    from main import load_config

IMPROVEMENT_THRESHOLD = 0.05
MLFLOW_EXPERIMENT_NAME = "anime-rbm-retraining"
_mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlruns"))
MLFLOW_TRACKING_URI = f"file:///{_mlruns_path.replace(os.sep, '/')}"

CURRENT_MODEL_PATH = os.path.join(OUTPUT_DIR, "rbm_best_model.pth")
CANDIDATE_MODEL_PATH = os.path.join(OUTPUT_DIR, "rbm_candidate_model.pth")
RETRAIN_METRICS_PATH = os.path.join(OUTPUT_DIR, "retrain_metrics.json")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def refresh_data_from_app() -> bool:
    """Refresh training data from app database."""
    print("Refreshing data from app database...")

    try:
        from data_pipeline.run_pipeline import step_collect_app, step_process, step_prepare

        print("[1/3] Collecting app user data...")
        if not step_collect_app(None):
            print("Warning: App data collection failed or returned no data")

        print("[2/3] Processing and unifying data...")
        if not step_process(None):
            print("Error: Data processing failed")
            return False

        print("[3/3] Preparing training data...")
        if not step_prepare(None):
            print("Error: Training data preparation failed")
            return False

        print("Data refresh completed successfully!")
        return True

    except ImportError as e:
        print(f"Error: Could not import data pipeline modules: {e}")
        return False
    except Exception as e:
        print(f"Error during data refresh: {e}")
        return False


def load_current_model_metrics() -> Optional[Dict[str, float]]:
    """Load metrics from current production model."""
    if os.path.exists(RETRAIN_METRICS_PATH):
        try:
            with open(RETRAIN_METRICS_PATH, 'r') as f:
                data = json.load(f)
                return data.get('current_model', {})
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def evaluate_model(
    model: RBM,
    train_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
    k: int,
    device: str
) -> Dict[str, float]:
    """Evaluate model and return precision, MAP, and NDCG metrics."""
    model.eval()
    model.to(device)
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)

    with torch.no_grad():
        precision, mean_ap, mean_ndcg = evaluate_at_k(
            model, train_tensor, test_tensor, k=k, device=device
        )

    return {
        f'precision@{k}': precision,
        f'map@{k}': mean_ap,
        f'ndcg@{k}': mean_ndcg
    }


def load_and_evaluate_current_model(
    n_visible: int,
    n_hidden: int,
    train_tensor: torch.Tensor,
    test_tensor: torch.Tensor,
    k: int,
    device: str
) -> Optional[Dict[str, float]]:
    """Load current production model and evaluate it."""
    if not os.path.exists(CURRENT_MODEL_PATH):
        print(f"No current model found at {CURRENT_MODEL_PATH}")
        return None

    print(f"Loading current model from {CURRENT_MODEL_PATH}...")
    current_model = RBM(n_visible=n_visible, n_hidden=n_hidden)
    current_model.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=device))

    print("Evaluating current model...")
    return evaluate_model(current_model, train_tensor, test_tensor, k, device)


def compare_models(
    current_metrics: Optional[Dict[str, float]],
    new_metrics: Dict[str, float],
    primary_metric: str = 'map@10'
) -> Tuple[bool, float]:
    """Compare new model against current and return (should_promote, improvement)."""
    if current_metrics is None:
        print("No current model to compare against - new model will be promoted")
        return True, float('inf')

    current_value = current_metrics.get(primary_metric, 0)
    new_value = new_metrics.get(primary_metric, 0)

    if current_value == 0:
        improvement = float('inf') if new_value > 0 else 0
    else:
        improvement = (new_value - current_value) / current_value

    print(f"\n{'Metric':<20} {'Current':>12} {'New':>12} {'Change':>12}")
    print("-" * 60)

    for metric in sorted(set(list(current_metrics.keys()) + list(new_metrics.keys()))):
        curr = current_metrics.get(metric, 0)
        new = new_metrics.get(metric, 0)
        change = ((new - curr) / curr * 100) if curr > 0 else float('inf')
        change_str = f"{change:+.2f}%" if change != float('inf') else "N/A"
        marker = " *" if metric == primary_metric else ""
        print(f"{metric:<20} {curr:>12.4f} {new:>12.4f} {change_str:>12}{marker}")

    print("-" * 60)
    print(f"Threshold: {IMPROVEMENT_THRESHOLD * 100:.1f}% | Actual: {improvement * 100:.2f}%")

    should_promote = improvement >= IMPROVEMENT_THRESHOLD
    status = "PASS - will promote" if should_promote else "FAIL - keeping current"
    print(f"Result: {status}")

    return should_promote, improvement


def save_metrics(
    current_metrics: Optional[Dict[str, float]],
    new_metrics: Dict[str, float],
    promoted: bool,
    improvement: float,
    config: Dict[str, Any]
) -> None:
    """Save retraining metrics to JSON file."""
    data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'current_model': current_metrics,
        'new_model': new_metrics,
        'promoted': promoted,
        'improvement': improvement,
        'improvement_threshold': IMPROVEMENT_THRESHOLD,
        'config': {
            'n_hidden': config['model']['n_hidden'],
            'learning_rate': config['model']['learning_rate'],
            'batch_size': config['model']['batch_size'],
            'epochs': config['model']['epochs'],
            'k': config['model']['k'],
            'holdout_ratio': config['data']['holdout_ratio'],
            'min_likes_user': config['data']['min_likes_user'],
            'min_likes_anime': config['data']['min_likes_anime']
        }
    }

    if promoted:
        data['current_model'] = new_metrics

    os.makedirs(os.path.dirname(RETRAIN_METRICS_PATH) or OUTPUT_DIR, exist_ok=True)
    with open(RETRAIN_METRICS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Metrics saved to {RETRAIN_METRICS_PATH}")


def promote_model() -> None:
    """Promote candidate model to production."""
    if os.path.exists(CANDIDATE_MODEL_PATH):
        if os.path.exists(CURRENT_MODEL_PATH):
            backup_path = CURRENT_MODEL_PATH.replace('.pth', '_backup.pth')
            shutil.copy2(CURRENT_MODEL_PATH, backup_path)
            print(f"Backed up current model to {backup_path}")

        shutil.move(CANDIDATE_MODEL_PATH, CURRENT_MODEL_PATH)
        print(f"Promoted candidate model to {CURRENT_MODEL_PATH}")
    else:
        print(f"Error: Candidate model not found at {CANDIDATE_MODEL_PATH}")


def retrain(
    refresh_data: bool = False,
    force_promote: bool = False,
    dry_run: bool = False
) -> bool:
    """Run full retraining pipeline with optional data refresh and model comparison."""
    print(f"Starting retraining at {datetime.now(timezone.utc).isoformat()}")
    print(f"Options: refresh_data={refresh_data}, force_promote={force_promote}, dry_run={dry_run}")

    if refresh_data:
        if not refresh_data_from_app():
            print("Warning: Data refresh failed, continuing with existing data")

    config = load_config()
    model_cfg = config['model']
    data_cfg = config['data']

    set_seed(SEED)

    print("\nLoading data...")
    ratings, anime = load_anime_dataset()
    user_anime, _ = preprocess_data(
        ratings,
        min_likes_user=data_cfg['min_likes_user'],
        min_likes_anime=data_cfg['min_likes_anime']
    )

    train_df, test_arr = make_train_test_split(
        user_anime,
        holdout_ratio=data_cfg['holdout_ratio']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.FloatTensor(train_df.values).to(device)
    test_tensor = torch.FloatTensor(test_arr).to(device)

    n_users, n_anime = train_tensor.shape
    print(f"Data loaded: {n_users} users, {n_anime} anime, device={device}")

    current_metrics = load_and_evaluate_current_model(
        n_visible=n_anime,
        n_hidden=model_cfg['n_hidden'],
        train_tensor=train_tensor,
        test_tensor=test_tensor,
        k=model_cfg['k'],
        device=str(device)
    )

    if current_metrics:
        k_val = model_cfg['k']
        print(f"Current model MAP@{k_val}: {current_metrics.get(f'map@{k_val}', 0):.4f}")

    print("\nTraining new model...")
    set_seed(SEED)
    new_model = RBM(n_visible=n_anime, n_hidden=model_cfg['n_hidden']).to(device)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'n_hidden': model_cfg['n_hidden'],
            'learning_rate': model_cfg['learning_rate'],
            'batch_size': model_cfg['batch_size'],
            'epochs': model_cfg['epochs'],
            'k': model_cfg['k'],
            'holdout_ratio': data_cfg['holdout_ratio'],
            'min_likes_user': data_cfg['min_likes_user'],
            'min_likes_anime': data_cfg['min_likes_anime'],
            'n_users': n_users,
            'n_anime': n_anime,
            'refresh_data': refresh_data,
            'force_promote': force_promote,
            'dry_run': dry_run
        })

        new_model, losses, precs, maps, ndcgs = train_rbm(
            new_model, train_tensor, test_tensor,
            epochs=model_cfg['epochs'],
            batch_size=model_cfg['batch_size'],
            learning_rate=model_cfg['learning_rate'],
            k=model_cfg['k'],
            device=str(device),
            use_mlflow=True
        )

        new_metrics = {
            f'precision@{model_cfg["k"]}': precs[-1] if precs else 0,
            f'map@{model_cfg["k"]}': maps[-1] if maps else 0,
            f'ndcg@{model_cfg["k"]}': ndcgs[-1] if ndcgs else 0
        }

        should_promote, improvement = compare_models(
            current_metrics,
            new_metrics,
            primary_metric=f'map@{model_cfg["k"]}'
        )

        mlflow.log_metrics({
            'improvement': improvement if improvement != float('inf') else 1.0,
            'should_promote': 1.0 if should_promote else 0.0,
            'promoted': 1.0 if (should_promote or force_promote) and not dry_run else 0.0
        })

        if current_metrics:
            for metric, value in current_metrics.items():
                mlflow.log_metric(f'current_{metric}', value)

    if dry_run:
        print("[DRY RUN] Skipping model promotion and metric saving")
        promoted = False
    elif force_promote:
        print("[FORCE] Promoting new model regardless of performance")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(new_model.state_dict(), CANDIDATE_MODEL_PATH)
        promote_model()
        promoted = True
    elif should_promote:
        print("Promoting new model...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(new_model.state_dict(), CANDIDATE_MODEL_PATH)
        promote_model()
        promoted = True
    else:
        print("Keeping current model (new model did not meet threshold)")
        promoted = False

    if not dry_run:
        save_metrics(current_metrics, new_metrics, promoted, improvement, config)

    k_val = model_cfg['k']
    print(f"\nCompleted at {datetime.now(timezone.utc).isoformat()}")
    print(f"New MAP@{k_val}: {new_metrics.get(f'map@{k_val}', 0):.4f} | Promoted: {promoted}")

    return True


def main():
    """CLI entry point for retraining."""
    parser = argparse.ArgumentParser(description="Automated model retraining")
    parser.add_argument('--refresh-data', action='store_true',
                        help='Refresh training data from app database')
    parser.add_argument('--force', action='store_true',
                        help='Force promote new model regardless of performance')
    parser.add_argument('--dry-run', action='store_true',
                        help='Train and evaluate but do not save or promote')

    args = parser.parse_args()
    success = retrain(
        refresh_data=args.refresh_data,
        force_promote=args.force,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
