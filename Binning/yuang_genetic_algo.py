import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import sample, seed
from scipy.optimize import differential_evolution

# Initializes seed
np.random.seed(1)
seed(32611)

# Loads data
try:
    filename = './Binning/pkls/data.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
except:
    exit(1)
try:
    filename = './Binning/pkls/pos_data.pkl'
    with open(filename, 'rb') as f:
        pos_data = pickle.load(f)
except:
    exit(1)
try:
    filename = './Binning/pkls/error_data.pkl'
    with open(filename, 'rb') as f:
        error_data = pickle.load(f)
    # Finds absolute value of all elements
    error_data = [abs(error) for error in error_data]
except:
    exit(1)


# Finds edge bounds
sorted_data = pos_data.copy()
sorted_data.sort()
B_LEFT = sorted_data[0]
B_RIGHT = sorted_data[len(sorted_data) - 1]
print(f'{B_LEFT}, {B_RIGHT}')


# Returns a subset of the data
# initial_data is the dataset you want to split
# percent is the percent of the initial_data you want returned
# total_size is the size of the initial data before any splitting
def reduce_data(initial_data, percent, total_size):
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

    indices = sample([num for num in range(int(len(initial_data['RealTrajectories'])))], int(total_size * percent))

    for index in indices:
        new_data['RealTrajectories'].append(initial_data['RealTrajectories'][index])
        new_data['EstTrajectories'].append(initial_data['EstTrajectories'][index])
        new_data['Errors'].append(initial_data['Errors'][index])

    indices.sort()

    for index in reversed(indices):
        initial_data['RealTrajectories'].pop(index)
        initial_data['EstTrajectories'].pop(index)
        initial_data['Errors'].pop(index)

    return new_data


# Splits data into subtrajectories for each region
def split_data(bounds, data, num_regions):
    bounds = list(bounds)
    full_bounds = [B_LEFT] + bounds + [B_RIGHT]

    trajectories = []
    for i in range(num_regions):
        trajectories.append([])

    for i in range(len(data['RealTrajectories'])):
        real_traj = data['RealTrajectories'][i]
        est_traj = data['EstTrajectories'][i]

        cur_real_traj = []
        cur_est_traj = []

        indices = np.argsort(real_traj)
        for p in range(len(indices)):
            cur_real_traj.append(real_traj[indices[p]])
            cur_est_traj.append(est_traj[indices[p]])

        for j in range(num_regions):
            sub_traj = []
            for k in range(len(cur_real_traj)):
                if cur_real_traj[k] > full_bounds[j] and cur_real_traj[k] <= full_bounds[j+1]:
                    sub_traj.append((cur_real_traj[k], abs(cur_real_traj[k] - cur_est_traj[k])))
            if len(sub_traj) > 0:
                trajectories[j].append(sub_traj)
    
    return trajectories


# Returns 99% conformal bound errors for each region
def find_errors(bounds, data, num_regions):
    trajectories = split_data(bounds, data, num_regions)
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
        index = int(np.ceil((len(region_data[i])) * (1 - (0.05 / num_regions))) - 1)
        if len(region_data[i]) >= 1:
            errors.append(region_data[i][index])
        else:
            errors.append(100000)
    
    return errors, region_sizes

"""    
# Generates bounds where (x: B1, y: B2) and B1 <= B2
def generate_points(num_bounds, num_regions):
    bounds_list = []

    for i in range(num_bounds):
        cur_bounds = []
        for j in range(num_regions - 1):
            new_B = np.random.uniform(B_LEFT, B_RIGHT)

            cur_bounds.append(new_B)
        cur_bounds.sort()
        bounds_list.append(cur_bounds)

    return bounds_list
"""

# Calculates the loss function for a given set of bounds
def calculate_loss(bounds, data, num_regions):
    errors, steps = find_errors(bounds, data, num_regions)

    # Previous loss function that uses width of bins
    #loss = ((point[0] - B0) * errors[0]) + ((point[1] - point[0]) * errors[1]) + ((B3 - point[1]) * errors[2])
    
    loss = 0
    total_steps = 0
    for i in range(num_regions):
        total_steps += steps[i]
    for i in range(num_regions):
        loss += (steps[i] / total_steps) * errors[i]
    
    return loss

"""
def generate_initial_bounds(data):
    if initial_bound_type == '1':
        points = generate_points(100)
        losses = []
        count = 0
        # Calculates and saves the loss for each set of bounds
        for point in points:
            print(point)
            losses.append(calculate_loss(point))
            count += 1
            print(count)

        # Finds the bounds with the min loss
        min_loss = losses[0]
        min_index = 0
        for i in range(len(losses)):
            if losses[i] < min_loss:
                min_loss = losses[i]
                min_index = i   

        return points[min_index], min_loss
    
    else:
        point = []
        for i in range(num_regions - 1):
            point.append(((1.8 / num_regions) * (i + 1)) - 1.2)

        return point, calculate_loss(point, data)
"""
"""
# could explore a bfs-style approach
def simulated_annealiing(point, data):
    s = point
    s_loss = calculate_loss(point, data)
    T = 1
    T_min = 0.0001
    alpha = 0.9

    while T > T_min:
        for i in range(10):
            s_new = []
            for j in range(len(s)):
                new_B = np.random.uniform(max(s[j] - (0.5 * T), B_LEFT), min(s[j] + (0.5 * T), B_RIGHT))
                s_new.append(new_B)
            
            s_new.sort()
            s_new_loss = calculate_loss(s_new, data)

            if s_new_loss < s_loss:
                s = s_new
                s_loss = s_new_loss
        
        T = T * alpha
        print(f'{T}, {T_min}')
    
    return s
"""

# Need to keep same structure below to ensure same results
# Splits data into two datasets
total_size = len(data['RealTrajectories'])
data_one = reduce_data(data, 0.5, total_size)
data_two = reduce_data(data, 0.5, total_size)
# alpha_data is for region tuning, bounds_data for finding errors
data_one_size = len(data_one['RealTrajectories'])
alpha_data = reduce_data(data_one, 0.25, data_one_size)
bounds_data = reduce_data(data_one, 0.75, data_one_size)
# Example implementation for method 2:
# data_two_size = len(data_two['RealTrajectories'])
# ____ = reduce(data_two, percent, data_two_size)
# ____ = reduce(data_two, percent, data_two_size)

def objective(region_bounds, N):
    # Check constraint: region_bounds must be in ascending order
    # If not, return a large penalty (here, 1e10)
    """
    if any(region_bounds[i] >= region_bounds[i+1] for i in range(len(region_bounds)-1)):
        return 9000000
    """
    region_bounds.sort()

    upper_bounds, number_traj = find_errors(region_bounds, alpha_data, N)

    total_weighted_penalty = 0
    # Adjust the denominator if needed (using number_traj[0] three times seems unusual)
    total = sum(number_traj)
    for i in range(N):
        weight = number_traj[i] / total
        penalty = upper_bounds[i] * 10
        total_weighted_penalty += weight * penalty

    return total_weighted_penalty

def partition_space(N):
    X_min, X_max = B_LEFT, B_RIGHT  # Fixed min and max bounds for partitioning

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


"""YUANGS CODE"""
output_file = "Binning/results_500data_95%.txt"

with open(output_file, "w") as f:
    f.write(f"Data Percentage: {0.25}\n")  # Save data percentage

    for number_regions in range(2, 8):  # Iterate from 3 to 7
        # Partition the space into regions
        regions = partition_space(number_regions)
        regions_flat = [item for sublist in regions for item in (sublist if isinstance(sublist, tuple) else [sublist])]

        regions_flat.sort()

        # Compute the total loss

        """
        # Save results to the file
        f.write(f"  Number of Regions: {number_regions}\n")
        f.write(f"  Revised Regions: {regions_flat}\n")
        f.write(f"  Total Loss: {total_loss}\n\n")
        """
        
        fixed_regions_flat = []
        for i in range(len(regions_flat)):
            if i % 2 == 0:
                fixed_regions_flat.append(regions_flat[i])
        
        fixed_regions_flat.pop(0)

        total_loss = calculate_loss(fixed_regions_flat, bounds_data, number_regions)
        cp_regions = find_errors(fixed_regions_flat, bounds_data, number_regions)[0]

        f.write(f'Number Regions: {number_regions}\n')
        for i in range(number_regions):
            if i == 0:
                f.write(f'[{B_LEFT}, {fixed_regions_flat[i]}]: {cp_regions[i]}\n')
            elif i == number_regions - 1:
                f.write(f'[{fixed_regions_flat[i - 1]}, {B_RIGHT}]: {cp_regions[i]}\n\n')
            else:
                f.write(f'[{fixed_regions_flat[i - 1]}, {fixed_regions_flat[i]}]: {cp_regions[i]}\n')   

        plt.title('Perception Error vs Position')
        plt.xlabel('Position')
        plt.ylabel('Perception Error')
        plt.scatter(pos_data, error_data, s=1)
        plt.axvline(x = B_LEFT, color = 'black', label = 'Edge')
        plt.axvline(x = B_RIGHT, color = 'black', label = 'Edge')  
        for i in range(len(fixed_regions_flat)):
            plt.axvline(x = fixed_regions_flat[i], color = 'r')    
        plt.legend(loc = 1, prop={'size': 6})
        plt.show()    

        # Print progress for debugging
        print(f"Completed data_percent = {0.25}, num_regions = {number_regions}")
        """
        print(f"Revised Regions: {regions_flat}")
        print(f"Total Loss: {total_loss}")
        """

print(f"Results saved to {output_file}.")