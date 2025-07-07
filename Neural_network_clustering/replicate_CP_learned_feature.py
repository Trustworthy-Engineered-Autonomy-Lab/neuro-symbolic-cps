import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


#### Replicate the results from the paper
#### Define the partition model
class PartitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # 2 groups
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)  # shape (n, 2)


def pinball_loss(q, s, alpha=0.1):
    # q: shape (m,)  → one quantile per group
    # s: shape (n, 1) → conformity scores
    # returns shape (n, m)
    n = s.size(0)
    m = q.size(0)
    q_expanded = q.unsqueeze(0).repeat(n, 1)  # (n, m)
    s_expanded = s.repeat(1, m)               # (n, m)

    loss = torch.where(s_expanded <= q_expanded,
                       alpha * (q_expanded - s_expanded),
                       (1 - alpha) * (s_expanded - q_expanded))
    return loss  # shape (n, m)

# Generate toy data
np.random.seed(42)
n = 1000
X = np.random.uniform(-1, 1, n)
Y = np.array([x + np.random.normal(0, 1 if x < 0 else 2) for x in X])

# Convert to tensors
x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# Pretrained model: f(x) = x
f_pred = x_tensor  # shape (n, 1)

# Conformity scores
S = torch.abs(y_tensor - f_pred)  # shape (n, 1)


model = PartitionNet()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

m = 2  # number of regions
q = torch.full((m,), 0.5, requires_grad=False)  # initial quantiles

alpha = 0.1
epochs = 100

for epoch in range(epochs):
    ### Step A: Update quantiles q for fixed h(x)
    with torch.no_grad():
        probs = model(x_tensor)  # (n, m)
        losses = pinball_loss(q, S, alpha)  # (n, m)

        weighted_losses = probs * losses  # (n, m)
        avg_losses = weighted_losses.mean(dim=0)  # (m,)

        # Use torch.quantile to find per-group quantiles from scores
        for i in range(m):
            weights_i = probs[:, i]
            sorted_s, idx = torch.sort(S.squeeze())
            cum_weights = torch.cumsum(weights_i[idx], dim=0)
            total = cum_weights[-1]
            threshold = (1 - alpha) * total
            idx_thresh = torch.searchsorted(cum_weights, threshold)
            q[i] = sorted_s[min(idx_thresh, len(sorted_s)-1)]  # quantile estimate

    ### Step B: Update model h(x) for fixed q
    optimizer.zero_grad()
    probs = model(x_tensor)  # (n, m)
    losses = pinball_loss(q, S, alpha)  # (n, m)
    loss = (probs * losses).mean()
    loss.backward()
    print("Gradient of classifier weights:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.6f}")
        else:
            print(f"{name} has no gradient")
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | q: {q.detach().numpy()}")
