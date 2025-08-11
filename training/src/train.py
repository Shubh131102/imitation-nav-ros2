import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import BCPolicy

def load_npz(path):
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    X /= X.max() if X.max() > 0 else 1.0
    return torch.from_numpy(X), torch.from_numpy(y)

def train(data_path="data/tb3_runs.npz", epochs=20, bs=256, lr=1e-3, out="policy.pt"):
    X, y = load_npz(data_path)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    model = BCPolicy(in_dim=X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep:02d} | loss {np.mean(losses):.4f}")

    torch.save(model.state_dict(), out)
    print("Saved model to", out)

if __name__ == "__main__":
    train()
