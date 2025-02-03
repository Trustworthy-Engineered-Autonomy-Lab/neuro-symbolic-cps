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
import pickle
import random
from random import sample, seed
# Initializes seed
seed(32611)

###Define the fixed dataset

# Loads data
def reduce_data(initial_data, percent, total_size):
    """Reduce the dataset by randomly sampling a given percentage."""

    new_data = {
        'RealTrajectories': [],
        'EstTrajectories': [],
        'Errors': []
    }

    if len(initial_data['RealTrajectories']) == 0:
        return new_data
    elif (total_size * percent) > (len(initial_data['RealTrajectories'])):
        print('Not enough data. Decrease percent value.')
        return new_data

    # Randomly select indices
    indices = random.sample(range(len(initial_data['RealTrajectories'])), int(total_size * percent))

    # Extract selected data
    for index in indices:
        new_data['RealTrajectories'].append(initial_data['RealTrajectories'][index])
        new_data['EstTrajectories'].append(initial_data['EstTrajectories'][index])
        new_data['Errors'].append(initial_data['Errors'][index])

    indices.sort()  # Sort indices to remove in reverse order
    print("Selected Indices:", indices)

    # Remove selected data from the original dataset
    for index in reversed(indices):
        initial_data['RealTrajectories'].pop(index)
        initial_data['EstTrajectories'].pop(index)
        initial_data['Errors'].pop(index)

    return new_data


# ###### old reduce_data function
# def reduce_data(percent):
#     if percent != 1.0:
#         total_size = len(data['RealTrajectories'])
#         count = 0
#         while (float(total_size - count) / total_size) > percent:
#             index = int(np.random.uniform(0, len(data['RealTrajectories'])))
#             data['RealTrajectories'].pop(index)
#             data['EstTrajectories'].pop(index)
#             data['Errors'].pop(index)
#             count += 1


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


def partition_space(N):
    X_min, X_max = -1.2, 0.6  # Fixed min and max bounds for partitioning

    # Define bounds for each region boundary
    bounds = [(X_min, X_max)] * (N - 1)

    # Differential Evolution Optimization
    result = differential_evolution(
        func=lambda region_bounds: objective(region_bounds, N),
        bounds=bounds,
        strategy='best1bin',
        maxiter=1000,
        tol=1e-7,
        disp=True
    )

    # Extract the optimized region boundaries
    final_bounds = [X_min] + list(result.x) + [X_max]
    regions = [(final_bounds[i], final_bounds[i + 1]) for i in range(N)]

    return regions

#### Old version of X
# def partition_space(X, N):
#     bounds = [(X.min(), X.max())] * (N - 1)  # Bounds for each region boundary
#
#     # Differential Evolution Optimization
#     result = differential_evolution(
#         func=lambda region_bounds: objective(region_bounds, N),
#         bounds=bounds,
#         strategy='best1bin',
#         maxiter=1000,
#         tol=1e-7,
#         disp=True
#     )
#
#     # Extract the optimized region boundaries
#     final_bounds = [X.min()] + list(result.x) + [X.max()]
#     regions = [(final_bounds[i], final_bounds[i + 1]) for i in range(N)]
#
#     return regions


# second version
def objective(region_bounds, N):
    # Check constraint: region_bounds must be in ascending order
    # If not, return a large penalty (here, 1e10)
    if any(region_bounds[i] >= region_bounds[i+1] for i in range(len(region_bounds)-1)):
        return 9000000

    upper_bounds, number_traj = find_errors(region_bounds)

    total_weighted_penalty = 0
    # Adjust the denominator if needed (using number_traj[0] three times seems unusual)
    total = sum(number_traj)
    for i in range(N):
        weight = number_traj[i] / total
        penalty = upper_bounds[i] * 10
        total_weighted_penalty += weight * penalty

    return total_weighted_penalty


def evaluate_loss(region_bounds, N):
    upper_bounds, number_traj = find_errors(region_bounds)

    total_weighted_penalty = 0
    # Adjust the denominator if needed (using number_traj[0] three times seems unusual)
    total = sum(number_traj)
    for i in range(N):
        weight = number_traj[i] / total
        penalty = upper_bounds[i] * 10
        total_weighted_penalty += weight * penalty

    return total_weighted_penalty


np.random.seed(32608)


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

# Ensure `data` is defined before using it
total_size = len(data['RealTrajectories'])

# First reduction: Split data into two parts
data_one = reduce_data(data, 0.5, total_size)
data_two = reduce_data(data, 0.5, total_size)

# Further split data_one
data_one_size = len(data_one['RealTrajectories'])  # Fixed missing closing bracket


output_file = "results_500data_95%.txt"
optimization_data = reduce_data(data_one, 0.25, data_one_size)  # Adjust the data based on data_percent
calculate_cp_data = reduce_data(data_one, 0.75, data_one_size)

with open(output_file, "w") as f:
    f.write(f"Data Percentage: {0.25}\n")  # Save data percentage

    for num_regions in range(3, 8):  # Iterate from 3 to 7
        # Partition the space into regions
        regions = partition_space(num_regions)
        regions_flat = [item for sublist in regions for item in (sublist if isinstance(sublist, tuple) else [sublist])]

        # Compute the total loss
        total_loss = evaluate_loss(regions_flat, num_regions)
        cp_regions = find_errors()
        # Save results to the file
        f.write(f"  Number of Regions: {num_regions}\n")
        f.write(f"  Revised Regions: {regions_flat}\n")
        f.write(f"  Total Loss: {total_loss}\n\n")

        # Print progress for debugging
        print(f"Completed data_percent = {0.25}, num_regions = {num_regions}")
        print(f"Revised Regions: {regions_flat}")
        print(f"Total Loss: {total_loss}")

print(f"Results saved to {output_file}.")



# ###############################################################
# ##### Test different ratio of the data for optimization #######
# output_file = "results.txt"
# data_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Percentages to iterate over
#
# with open(output_file, "w") as f:
#     for data_percent in data_percentages:  # Iterate over data percentages
#         reduce_data(data_percent)  # Adjust the data based on data_percent
#         f.write(f"Data Percentage: {data_percent}\n")  # Save data percentage
#
#         for num_regions in range(3, 8):  # Iterate from 3 to 7
#             X = np.array(pos_data)  # Position data
#
#             # Partition the space into regions
#             regions = partition_space(X, num_regions)
#             regions_flat = [item for sublist in regions for item in (sublist if isinstance(sublist, tuple) else [sublist])]
#
#             # Compute the total loss
#             total_loss = evaluate_loss(regions_flat, num_regions)
#
#             # Save results to the file
#             f.write(f"  Number of Regions: {num_regions}\n")
#             f.write(f"  Revised Regions: {regions_flat}\n")
#             f.write(f"  Total Loss: {total_loss}\n\n")
#
#             # Print progress for debugging
#             print(f"Completed data_percent = {data_percent}, num_regions = {num_regions}")
#             print(f"Revised Regions: {regions_flat}")
#             print(f"Total Loss: {total_loss}")
#
# print(f"Results saved to {output_file}.")
