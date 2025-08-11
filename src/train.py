import torch

from constants import (
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_K,
    WEIGHT_DECAY, CLAMP_MIN, CLAMP_MAX, EARLY_STOPPING_PATIENCE
)
from src.evaluate import evaluate_at_k

def train_single_batch(rbm, optimizer, batch):
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

    loss = torch.mean((v0 - vk) ** 2)
    return loss.item()

def evaluate_and_log(rbm, train_tensor, test_tensor, k, device, epoch, epoch_loss):
    rbm.eval()
    with torch.no_grad():
        precision, mean_ap, mean_ndcg = evaluate_at_k(rbm, train_tensor, test_tensor, k=k, device=device)
    
    print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | "
          f"Precision@{k}: {precision:.4f} | MAP@{k}: {mean_ap:.4f} | NDCG@{k}: {mean_ndcg:.4f}")
    
    return precision, mean_ap, mean_ndcg

def check_early_stopping(mean_ap, best_map, patience_counter, patience, rbm):
    if mean_ap > best_map:
        best_map = mean_ap
        patience_counter = 0
        best_model_state = rbm.state_dict().copy()
    else:
        patience_counter += 1
        should_stop = patience_counter >= patience
        if should_stop:
            print("Early stopping triggered.")
        best_model_state = None
    
    return best_map, patience_counter, should_stop, best_model_state

def save_best_model(rbm, best_model_state, best_map, k):
    if best_model_state is not None:
        rbm.load_state_dict(best_model_state)
        quantized_rbm = torch.quantization.quantize_dynamic(
            rbm, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_rbm.state_dict(), "out/rbm_best_model.pth")
        print(f"Best model saved with MAP@{k}: {best_map:.4f}")

def train_rbm(rbm, train_tensor, test_tensor, 
              epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, 
              learning_rate=DEFAULT_LEARNING_RATE, k=DEFAULT_K, device='cpu'):
    rbm.to(device)
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)

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
            batch = train_tensor[i:i+batch_size]
            batch_loss = train_single_batch(rbm, optimizer, batch)
            epoch_loss += batch_loss
            num_batches += 1

        epoch_loss /= num_batches
        losses.append(epoch_loss)

        precision, mean_ap, mean_ndcg = evaluate_and_log(
            rbm, train_tensor, test_tensor, k, device, epoch, epoch_loss
        )
        precs.append(precision)
        maps.append(mean_ap)
        ndcgs.append(mean_ndcg)

        best_map, patience_counter, should_stop, model_state = check_early_stopping(
            mean_ap, best_map, patience_counter, EARLY_STOPPING_PATIENCE, rbm
        )
        if model_state is not None:
            best_model_state = model_state
        if should_stop:
            break

    save_best_model(rbm, best_model_state, best_map, k)
    return rbm, losses, precs, maps, ndcgs
