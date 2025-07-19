import torch
import torch.nn as nn
from src.base_model import BaseRecommender
from src.evaluate import evaluate_at_k

class RBM(nn.Module, BaseRecommender):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.xavier_uniform_(self.W)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def fit(self, train_tensor, test_tensor, epochs=20, batch_size=32, learning_rate=0.001, k=10, device='cpu', **kwargs):
        self.to(device)
        train_tensor = train_tensor.to(device)
        test_tensor = test_tensor.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        losses, precs, maps, ndcgs = [], [], [], []

        best_map = 0.0
        patience = 3
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, train_tensor.size(0), batch_size):
                v0 = train_tensor[i:i+batch_size]
                ph0, _ = self.sample_h(v0)
                _, vk = self.sample_v(ph0)
                phk, _ = self.sample_h(vk)
                vk = torch.clamp(vk, 0.0, 1.0)

                positive_grad = torch.bmm(ph0.unsqueeze(2), v0.unsqueeze(1))
                negative_grad = torch.bmm(phk.unsqueeze(2), vk.unsqueeze(1))

                dW = (positive_grad - negative_grad).mean(0)
                dv_bias = (v0 - vk).mean(0)
                dh_bias = (ph0 - phk).mean(0)

                self.W.grad = -dW
                self.v_bias.grad = -dv_bias
                self.h_bias.grad = -dh_bias

                optimizer.step()
                optimizer.zero_grad()

                loss = torch.mean((v0 - vk) ** 2)
                epoch_loss += loss.item()
                num_batches += 1

            epoch_loss /= num_batches
            losses.append(epoch_loss)

            # Evaluation
            self.eval()
            with torch.no_grad():
                precision, mean_ap, mean_ndcg = evaluate_at_k(self, train_tensor, test_tensor, k=k, device=device)
                precs.append(precision)
                maps.append(mean_ap)
                ndcgs.append(mean_ndcg)

            if mean_ap > best_map:
                best_map = mean_ap
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | "
                  f"Precision@{k}: {precision:.4f} | MAP@{k}: {mean_ap:.4f} | NDCG@{k}: {mean_ndcg:.4f}")

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return {
            "losses": losses,
            "precs": precs,
            "maps": maps,
            "ndcgs": ndcgs
        }

    def predict(self, user_tensor, device='cpu'):
            self.eval()
            user_tensor = user_tensor.to(device)
            with torch.no_grad():
                p_h, _ = self.sample_h(user_tensor)
                p_v, _ = self.sample_v(p_h)
            return p_v

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()

    def sample_h(self, v):
        p_h = torch.sigmoid(torch.mm(v, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        p_v = torch.sigmoid(torch.mm(h, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)
