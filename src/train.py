import torch
from src.evaluate import evaluate_at_k

def train_rbm(rbm, train_tensor, test_tensor, 
              epochs=20, batch_size=32, learning_rate=0.001, k=10, device='cpu'):
    rbm.to(device)
    train_tensor = train_tensor.to(device)
    test_tensor = test_tensor.to(device)

    optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate, weight_decay=1e-5)

    losses, precs, maps, ndcgs = [], [], [], []

    best_map = 0.0
    patience = 3
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        rbm.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, train_tensor.size(0), batch_size):
            v0 = train_tensor[i:i+batch_size]
            ph0, _ = rbm.sample_h(v0)
            _, vk = rbm.sample_v(ph0)
            phk, _ = rbm.sample_h(vk)
            vk = torch.clamp(vk, 0.0, 1.0)

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
            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        losses.append(epoch_loss)

        # Evaluation
        rbm.eval()
        with torch.no_grad():
            precision, mean_ap, mean_ndcg = evaluate_at_k(rbm, train_tensor, test_tensor, k=k, device=device)
            precs.append(precision)
            maps.append(mean_ap)
            ndcgs.append(mean_ndcg)

        if mean_ap > best_map:
            best_map = mean_ap
            patience_counter = 0
            best_model_state = rbm.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | "
            f"Precision@{k}: {precision:.4f} | MAP@{k}: {mean_ap:.4f} | NDCG@{k}: {mean_ndcg:.4f}")

    if best_model_state is not None:
        torch.save(best_model_state, "rbm_best_model.pth")
        print(f"Best model saved with MAP@{k}: {best_map:.4f}")
    return rbm, losses, precs, maps, ndcgs
