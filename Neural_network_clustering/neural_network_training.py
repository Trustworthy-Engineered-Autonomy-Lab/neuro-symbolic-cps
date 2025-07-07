import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

# Load data
with open('binning_data/data.pkl', 'rb') as f:
    data = pickle.load(f)

# Determine position bounds
all_positions = [p for traj in data['RealTrajectories'] for p in traj]
B_LEFT, B_RIGHT = min(all_positions), max(all_positions)

# --- Split and Find Errors ---
def split_data(bounds, data, num_regions):
    bounds = list(bounds)
    full_bounds = [B_LEFT] + bounds + [B_RIGHT]
    trajectories = [[] for _ in range(num_regions)]

    for real, est in zip(data['RealTrajectories'], data['EstTrajectories']):
        indices = np.argsort(real)
        sorted_real = [real[i] for i in indices]
        sorted_est = [est[i] for i in indices]

        for j in range(num_regions):
            sub_traj = [(sorted_real[k], abs(sorted_real[k] - sorted_est[k]))
                        for k in range(len(sorted_real))
                        if full_bounds[j] < sorted_real[k] <= full_bounds[j+1]]
            if sub_traj:
                trajectories[j].append(sub_traj)

    return trajectories

def find_errors(bounds, data, num_regions):
    trajectories = split_data(bounds, data, num_regions)
    errors, sizes = [], []

    for region in trajectories:
        max_errors = [max(e for _, e in traj) for traj in region if traj]
        if max_errors:
            max_errors.sort()
            q_index = int(np.ceil(len(max_errors) * (1 - 0.05 / num_regions))) - 1
            errors.append(max_errors[q_index])
        else:
            errors.append(100000)
        sizes.append(len(max_errors))
    return errors, sizes

# --- Dataset Generation ---
def generate_supervised_dataset(N, data, num_regions=3):
    X, y = [], []
    for _ in range(N):
        b1 = random.uniform(B_LEFT, B_RIGHT - 0.01)
        b2 = random.uniform(b1 + 0.01, B_RIGHT)
        bounds = [b1, b2]
        errors, sizes = find_errors(bounds, data, num_regions)
        total = sum(sizes)
        if total == 0:
            continue
        weighted_loss = sum((sizes[i] / total) * errors[i] for i in range(num_regions))
        X.append([b1, b2])
        y.append(weighted_loss)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- Neural Network ---
class RegionLossRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- Training ---
model = RegionLossRegressor()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train, y_train = generate_supervised_dataset(100000, data)
torch.save((X_train, y_train), 'region_training_dataset2.pt')

for epoch in range(300):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# --- Optimize [b1, b2] After Training ---
b = torch.tensor([0.3, 0.6], requires_grad=True)
b_opt = optim.Adam([b], lr=0.01)

for step in range(300):
    b_opt.zero_grad()
    b_sorted = torch.sort(b)[0]
    pred_loss = model(b_sorted.unsqueeze(0))
    pred_loss.backward()
    b_opt.step()
    if step % 50 == 0:
        print(f"Step {step} | Predicted Loss: {pred_loss.item():.4f}")

b1_opt, b2_opt = b_sorted.detach().tolist()
print(f"🔍 Optimized Bounds: b1 = {b1_opt:.4f}, b2 = {b2_opt:.4f}")
