"""RBM training with contrastive divergence and early stopping."""
import os
import sys
import torch
import mlflow as mlf
import mlflow.pytorch

try:
    from ..constants import (
        CLAMP_MIN, CLAMP_MAX, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, OUTPUT_DIR
    )
except ImportError:
    _CURRENT_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
    if _PROJECT_ROOT not in sys.path:
        sys.path.append(_PROJECT_ROOT)
    from constants import (
        CLAMP_MIN, CLAMP_MAX, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, OUTPUT_DIR
    )
from .evaluate import evaluate_at_k


def train_single_batch(rbm, optimizer, batch):
    """Train RBM on single batch using contrastive divergence."""
    v0 = batch
    ph0, _ = rbm.sample_h(v0)
    _, vk = rbm.sample_v(ph0)
    phk, _ = rbm.sample_h(vk)
    vk = torch.clamp(vk, CLAMP_MIN, CLAMP_MAX)

    positive_grad = torch.bmm(ph0.unsqueeze(2), v0.unsqueeze(1))
    negative_grad = torch.bmm(phk.unsqueeze(2), vk.unsqueeze(1))

    dW = (positive_grad - negative_grad).mean(0)
    dv_bias = (v0 - vk).mean(0)
    dh_bias = (ph0 - phk).mean(0)

    rbm.W.grad = -dW
    rbm.v_bias.grad = -dv_bias
    rbm.h_bias.grad = -dh_bias

    optimizer.step()
    optimizer.zero_grad()
    return torch.mean((v0 - vk) ** 2).item()


def evaluate_and_log(rbm, train_tensor, test_tensor, k, device, epoch, epoch_loss, use_mlflow=True):
    """Evaluate model and log metrics for current epoch."""
    rbm.eval()
    with torch.no_grad():
        precision, mean_ap, mean_ndcg = evaluate_at_k(rbm, train_tensor, test_tensor, k=k, device=device)

    print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Precision@{k}: {precision:.4f} | MAP@{k}: {mean_ap:.4f} | NDCG@{k}: {mean_ndcg:.4f}")

    if use_mlflow and mlf.active_run():
        mlf.log_metrics({
            "training_loss": epoch_loss,
            f"precision_{k}": precision,
            f"map_{k}": mean_ap,
            f"ndcg_{k}": mean_ndcg
        }, step=epoch + 1)

    return precision, mean_ap, mean_ndcg


def check_early_stopping(mean_ap, best_map, patience_counter, patience, rbm):
    """Check if training should stop and save best model state."""
    if mean_ap > best_map:
        best_map = mean_ap
        patience_counter = 0
        best_model_state = {k: v.clone() for k, v in rbm.state_dict().items()}
        should_stop = False
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience
        if should_stop:
            print("Early stopping triggered.")
        best_model_state = None
    return best_map, patience_counter, should_stop, best_model_state


def save_best_model(rbm, best_model_state, best_map, k, model_path: str | None = None):
    """Save best model checkpoint to disk and return the path."""
    if best_model_state is None:
        return None
    rbm.load_state_dict(best_model_state)
    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, "rbm_best_model.pth")
    elif not os.path.isabs(model_path):
        model_path = os.path.join(OUTPUT_DIR, os.path.basename(model_path))
    os.makedirs(os.path.dirname(model_path) or OUTPUT_DIR, exist_ok=True)
    torch.save(rbm.state_dict(), model_path)
    print(f"Best model saved with MAP@{k}: {best_map:.4f} -> {model_path}")
    return model_path


def train_rbm(rbm, train_tensor, test_tensor,
              epochs=30, batch_size=32,
              learning_rate=0.001, k=10, device='cpu',
              use_mlflow=True):
    """Train RBM with MLflow logging and early stopping."""
    rbm.to(device)
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)

    if use_mlflow and mlf.active_run():
        mlf.log_params({
            "n_visible": rbm.W.shape[1],
            "n_hidden": rbm.W.shape[0],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "k": k,
            "device": str(device),
            "clamp_range": f"[{CLAMP_MIN}, {CLAMP_MAX}]"
        })
        mlf.set_tags({
            "model_type": "RBM",
            "train_users": train_tensor.shape[0],
            "n_anime": train_tensor.shape[1]
        })

    optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    losses, precs, maps, ndcgs = [], [], [], []
    best_map = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        rbm.train()
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, train_tensor.size(0), batch_size):
            batch = train_tensor[i:i + batch_size]
            epoch_loss += train_single_batch(rbm, optimizer, batch)
            num_batches += 1
        epoch_loss /= num_batches
        losses.append(epoch_loss)

        precision, mean_ap, mean_ndcg = evaluate_and_log(
            rbm, train_tensor, test_tensor, k, device, epoch, epoch_loss,
            use_mlflow=use_mlflow
        )
        precs.append(precision)
        maps.append(mean_ap)
        ndcgs.append(mean_ndcg)

        best_map, patience_counter, should_stop, model_state = check_early_stopping(
            mean_ap, best_map, patience_counter, EARLY_STOPPING_PATIENCE, rbm
        )
        if model_state is not None:
            best_model_state = model_state
            if use_mlflow and mlf.active_run():
                mlf.log_metric("best_map_so_far", best_map, step=epoch + 1)

        if should_stop:
            break

    model_path = save_best_model(rbm, best_model_state, best_map, k)

    if use_mlflow and mlf.active_run():
        mlf.log_metrics({
            "final_map": maps[-1] if maps else 0.0,
            "final_precision": precs[-1] if precs else 0.0,
            "final_ndcg": ndcgs[-1] if ndcgs else 0.0,
            "best_map": best_map,
            "total_epochs_trained": len(losses),
            "final_loss": losses[-1] if losses else 0.0
        })

        if model_path and os.path.exists(model_path):
            mlf.log_artifact(model_path, artifact_path="model")

        if best_model_state is not None:
            rbm.load_state_dict(best_model_state)
            mlf.pytorch.log_model(rbm, "pytorch_model")

        print(f"MLflow run ID: {mlf.active_run().info.run_id}")

    return rbm, losses, precs, maps, ndcgs
