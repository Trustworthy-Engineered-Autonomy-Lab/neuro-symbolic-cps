import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Step 1: Simulate Data ----
np.random.seed(42)
#torch.manual_seed(42)

n = 1000
X = np.random.uniform(-1, 1, n)
Y = np.array([x + np.random.normal(0, 1 if x < 0 else 2) for x in X])

x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
f_pred = x_tensor  # f(x) = x
S = torch.abs(y_tensor - f_pred)  # conformity scores

alpha = 0.1
target_coverage = 1 - alpha

# ---- Step 2: Define Region Classifier h(x) ----
class RegionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)  # shape (n, 2)

model = RegionClassifier()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# ---- Step 3: Differentiable Training Loop ----
for epoch in range(200):
    probs = model(x_tensor)  # (n, 2)
    total_loss = 0

    for i in range(2):  # loop over the 2 groups
        weights = probs[:, i]  # (n,)
        if weights.sum() == 0:
            continue

        # Sort conformity scores and weights for quantile estimation
        s_sorted, idx = torch.sort(S.squeeze())
        w_sorted = weights[idx]
        cum_weights = torch.cumsum(w_sorted, dim=0)
        threshold = (1 - alpha) * w_sorted.sum()

        # Quantile index
        q_idx = torch.searchsorted(cum_weights, threshold).clamp(max=len(s_sorted)-1)
        q = s_sorted[q_idx].detach()  # scalar, detached to avoid gradient through quantile

        # Soft coverage estimation using differentiable indicator
        s_i = S.squeeze()
        h_i = probs[:, i]
        #soft_indicators = (s_i <= q).float()
        soft_indicators = torch.sigmoid((q - s_i) * 100)
        coverage = (soft_indicators * h_i).sum() / (h_i.sum() + 1e-8)

        penalty = torch.relu(target_coverage - coverage) * 100  # enforce coverage constraint
        total_loss += q + penalty

    optimizer.zero_grad()
    total_loss.backward()
    print("Gradient of classifier weights:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm().item():.6f}")
        else:
            print(f"{name} has no gradient")
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f}")

# ---- Step 4: Predict interval for a new point ----
def predict_interval(x_val, model, alpha=0.1):
    x = torch.tensor([[x_val]], dtype=torch.float32)
    probs = model(x).squeeze()
    group = torch.argmax(probs).item()

    with torch.no_grad():
        all_probs = model(x_tensor)
        mask = all_probs[:, group] > 0.5
        s_group = S[mask]
        if s_group.numel() > 0:
            q = torch.quantile(s_group, 1 - alpha)
        else:
            q = torch.tensor(0.0)
        return x_val - q.item(), x_val + q.item()

# Example usage
x_val = 0.5
interval = predict_interval(x_val, model)
print(f"Prediction interval for x={x_val:.2f}: [{interval[0]:.2f}, {interval[1]:.2f}]")
