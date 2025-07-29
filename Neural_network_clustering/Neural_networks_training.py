
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import sys
from collections import Counter

def split_data(bounds):
    if isinstance(bounds, np.ndarray):
        full_bounds = [B_LEFT] + bounds.tolist() + [B_RIGHT]
    elif isinstance(bounds, list):
        full_bounds = [B_LEFT] + bounds + [B_RIGHT]
    else:
        raise TypeError("Bounds must be either a list or a numpy ndarray.")

    trajectories = []
    for i in range(num_regions):
        trajectories.append([])

    for i in range(len(optimization_data['RealTrajectories'])):
        cur_real_traj = optimization_data['RealTrajectories'][i]
        cur_est_traj = optimization_data['EstTrajectories'][i]
        cur_real_traj.sort()
        cur_est_traj.sort()

        for j in range(num_regions):
            sub_traj = []
            for k in range(len(cur_real_traj)):
                if cur_real_traj[k] > full_bounds[j] and cur_real_traj[k] <= full_bounds[j + 1]:
                    sub_traj.append((cur_real_traj[k], abs(cur_real_traj[k] - cur_est_traj[k])))
            trajectories[j].append(sub_traj)

    return trajectories


def find_errors(bounds):
    trajectories = split_data(bounds)

    region_data = []
    errors = []
    region_sizes = []

    for i in range(num_regions):
        size = 0
        temp_data = []
        for j in range(len(trajectories[i])):
            cur_max = 0
            for k in range(len(trajectories[i][j])):
                if trajectories[i][j][k][1] > cur_max:
                    cur_max = trajectories[i][j][k][1]
                size += 1
            temp_data.append(cur_max)
        temp_data.append(10000)
        temp_data.sort()
        region_data.append(temp_data)
        region_sizes.append(size)

    for i in range(num_regions):
        index = int(np.ceil((len(region_data[i])) * (1-0.05/num_regions)) - 1)
        if len(region_data[i]) >= 1:
            errors.append(region_data[i][index])
        else:
            errors.append(10000)

    return errors, region_sizes

#### Start the conversion  ####

# Finds edge bounds
# Loads data
try:
    filename = 'binning_data/data.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
except:
    exit(1)

try:
    filename = 'binning_data/pos_data.pkl'
    with open(filename, 'rb') as f:
        pos_data = pickle.load(f)
except:
    exit(1)
try:
    filename = 'binning_data/error_data.pkl'
    with open(filename, 'rb') as f:
        error_data = pickle.load(f)
    error_data = [abs(error) for error in error_data]
except:
    exit(1)


# Finds edge bounds
B_LEFT = -1.2
B_RIGHT = 0.6
print(f'{B_LEFT}, {B_RIGHT}')

## Generate Supervised Dataset
import random

def generate_supervised_dataset(num_samples, B_LEFT, B_RIGHT, data):
    X = []
    y = []

    for _ in range(num_samples):
        b1 = random.uniform(B_LEFT, B_RIGHT)
        b2 = random.uniform(b1 + 0.01, B_RIGHT)  # Ensure b1 < b2

        bounds = [b1, b2]
        errors, sizes = find_errors(bounds, data, 3)

        total = sum(sizes)
        if total == 0:
            continue

        weighted_error = sum((sizes[i] / total) * errors[i] for i in range(3))

        X.append([b1, b2])
        y.append(weighted_error)

    return X, y

## Train Neural Network to Approximate Weighted Error
import torch
import torch.nn as nn

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


X, y = generate_supervised_dataset(100, B_LEFT, B_RIGHT, data)
epochs = 20
model = RegionLossRegressor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(epochs):
    optimizer.zero_grad()
    preds = model(X_tensor)
    loss = loss_fn(preds, y_tensor)
    loss.backward()
    optimizer.step()






#### Small test ####
# # Ensure `data` is defined before using it
# total_size = len(data['RealTrajectories'])
# print(total_size)
#
# optimization_data = data
# num_regions = 2
#
# error, region_size = find_errors([0.22696756558551945])
# print(error)
#
#
# x_vals = np.linspace(-1.2, 0.6, 30000)
# min_weighted_error = float('inf')
# best_bound = None
# best_region_sizes = None
#
# for i in range(1, len(x_vals) - 1):
#     bounds = [x_vals[i]]
#     error, region_size = find_errors(bounds)
#
#     total_size = region_size[0] + region_size[1]
#     weighted_error = (error[0] * region_size[0] + error[1] * region_size[1]) / total_size
#
#     if weighted_error < min_weighted_error:
#         min_weighted_error = weighted_error
#         best_bound = bounds
#         best_region_sizes = region_size
#
# print("Minimal Weighted Error:", min_weighted_error)
# print("Best Bound:", best_bound)
# print("Total Region Size:", best_region_sizes[0] + best_region_sizes[1])
#
#
#
# ##### quick test #####
# bounds = [0.2]
# error, region_size = find_errors(bounds)
# weighted_error = error[0] * region_size[0]/(region_size[0] + region_size[1]) + error[1] * region_size[1]/(region_size[0] + region_size[1])
# print(weighted_error)
# print(region_size[0] + region_size[1])
#1. Quick test for simmulated annealing


# 4000 dataset and split into 2 parts



# Calculate the bounds for each part



# Do it in a for loop



