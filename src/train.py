import torch

def train_rbm(rbm, train_tensor, batch_size=32, epochs=20, lr=0.001, verbose=True):

    #optimizer = torch.optim.SGD(rbm.parameters(), lr=lr)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        rbm.train()
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, train_tensor.size(0), batch_size):
            optimizer.zero_grad()

            v0 = train_tensor[i:i+batch_size]
            ph0, _ = rbm.sample_h(v0)
            
            # CD-1
            vk = v0.clone().detach()
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            phk, _ = rbm.sample_h(vk)
            
            batch_size = v0.size(0)
            rbm.W.grad     = -(ph0.t() @ v0 - phk.t() @ vk) / batch_size
            rbm.v_bias.grad = -(v0 - vk).mean(dim=0)
            rbm.h_bias.grad = -(ph0 - phk).mean(dim=0)

            optimizer.step()

            loss = torch.mean((v0 - vk) ** 2)
            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss = epoch_loss / num_batches

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
