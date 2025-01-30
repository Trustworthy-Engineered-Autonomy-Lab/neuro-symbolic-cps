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


def reduce_data(percent):
    if percent != 1.0:
        total_size = len(data['RealTrajectories'])
        count = 0
        while (float(total_size - count) / total_size) > percent:
            index = int(np.random.uniform(0, len(data['RealTrajectories'])))
            data['RealTrajectories'].pop(index)
            data['EstTrajectories'].pop(index)
            data['Errors'].pop(index)
            count += 1


def split_data(bounds):
    print(type(bounds))
    full_bounds = [B_LEFT] + bounds.tolist() + [B_RIGHT]
    #full_bounds = [B_LEFT] + bounds + [B_RIGHT]
    trajectories = []
    for i in range(num_regions):
        trajectories.append([])

    for i in range(len(data['RealTrajectories'])):
        cur_real_traj = data['RealTrajectories'][i]
        cur_est_traj = data['EstTrajectories'][i]
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
        temp_data.append(100000)
        temp_data.sort()
        region_data.append(temp_data)
        region_sizes.append(size)

    for i in range(num_regions):
        index = int(np.ceil((len(region_data[i])) * 0.99) - 1)
        if len(region_data[i]) >= 1:
            errors.append(region_data[i][index])
        else:
            errors.append(100000)

    return errors, region_sizes


##### Previous version #####
# def split_data(B1, B2):
#     bounds = [B0, B1, B2, B3]
#     trajectories = [[],[],[]]
#     for i in range(len(data['RealTrajectories'])):
#         flag = -1
#         for j in range(len(data['RealTrajectories'][i])):
#             if flag == -1:
#                 if j == 0:
#                     temp_traj = [(data['RealTrajectories'][i][j], data['Errors'][i][j])]
#                 if temp_traj[0][0] < bounds[1]:
#                     flag = 1
#                 elif temp_traj[0][0] < bounds[2]:
#                     flag = 2
#                 elif temp_traj[0][0] < bounds[3]:
#                     flag = 3
#             else:
#                 if data['RealTrajectories'][i][j] <= bounds[flag] and data['RealTrajectories'][i][j] >= bounds[flag - 1]:
#                     temp_traj.append((data['RealTrajectories'][i][j], data['Errors'][i][j]))
#                     if j == len(data['RealTrajectories'][i]) - 1:
#                         trajectories[flag - 1].append(temp_traj)
#                 else:
#                     trajectories[flag - 1].append(temp_traj)
#                     flag = -1
#                     temp_traj = [(data['RealTrajectories'][i][j], data['Errors'][i][j])]
#     return trajectories
#
#
# def find_errors(B1, B2):
#     trajectories = split_data(B1, B2)
#     region_data = [[],[],[]]
#     errors = [0,0,0]
#     region_size = [0,0,0]
#     for i in range(3):
#         temp_data = []
#         for j in range(len(trajectories[i])):
#             cur_max = 0
#             for k in range(len(trajectories[i][j])):
#                 if trajectories[i][j][k][1] > cur_max:
#                     cur_max = trajectories[i][j][k][1]
#                 region_size[i] += 1
#             temp_data.append(cur_max)
#         temp_data.append(10)
#         temp_data.sort()
#         region_data[i] = temp_data
#     for i in range(3):
#         index = int(np.ceil((len(region_data[i])) * 0.99) - 1)
#         if len(region_data[i]) >= 1:
#             errors[i] = region_data[i][index]
#         else:
#             errors[i] = 0
#     return errors, region_size



def partition_space(X, N):
    bounds = [(X.min(), X.max())] * (N - 1)  # Bounds for each region boundary

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
    final_bounds = [X.min()] + list(result.x) + [X.max()]
    regions = [(final_bounds[i], final_bounds[i + 1]) for i in range(N)]

    return regions


# second version
def objective(region_bounds, N):
    # Check constraint: region_bounds must be in ascending order
    # If not, return a large penalty (here, 1e10)
    if any(region_bounds[i] >= region_bounds[i+1] for i in range(len(region_bounds)-1)):
        return 1e10
    # old version
    # If boundaries are valid, continue with the evaluation
    # bounds_numbers = find_errors(region_bounds[0], region_bounds[1])
    # upper_bounds = bounds_numbers[0]
    # number_traj = bounds_numbers[1]
    # New version
    upper_bounds, number_traj = find_errors(region_bounds)

    total_weighted_penalty = 0
    # Adjust the denominator if needed (using number_traj[0] three times seems unusual)
    total = sum(number_traj)
    for i in range(N):
        weight = number_traj[i] / total
        penalty = upper_bounds[i] * 10
        total_weighted_penalty += weight * penalty

    return total_weighted_penalty


np.random.seed(1)

# data_percent = float(input('Enter percent of data to use (0.0,1.0): '))
# num_regions = int(input('Enter number of regions: '))
data_percent = 0.3
num_regions = 3

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

sorted_data = pos_data.copy()
sorted_data.sort()
B_LEFT = sorted_data[0]
B_RIGHT = sorted_data[len(sorted_data) - 1]
print(f'{B_LEFT}, {B_RIGHT}')


reduce_data(data_percent) # change the dat

X = np.array(pos_data)  # Position data
# Partition the space and get the regions
regions = partition_space(X, num_regions)
print('here are all the split regions:', regions)









#############################################################################################################
# Loads data
try:
    filename = 'binning_data/Thomas_all_data.pkl'
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
except:
    exit(1)
# Finds edge bounds
sorted_data = pos_data.copy()
sorted_data.sort()
B0 = sorted_data[0]
B3 = sorted_data[len(sorted_data) - 1]
print('here is the upper and lower bounds:', f'{B0}, {B3}')







# Run the global algorithm
X = np.array(pos_data)  # Position data
f_X = np.abs(np.array(error_data))
N = 3  # Number of regions (you can change this based on your requirement)
# Partition the space and get the regions
regions = partition_space(X, f_X, N)
print('here are all the split regions:', regions)
all_bounds = np.array(regions).flatten()
counts = Counter(all_bounds)
two_bounds = [value for value, count in counts.items() if count > 1]
errors_number_of_traj = find_errors(two_bounds[0], two_bounds[1])
CP_bounds_99 = errors_number_of_traj[0]
print(CP_bounds_99)