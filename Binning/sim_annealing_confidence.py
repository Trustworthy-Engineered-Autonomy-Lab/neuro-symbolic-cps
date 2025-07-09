import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import sample, seed
from scipy.optimize import differential_evolution

# TO DO:
# check every calculate loss and find error function and include confidence values


# Initializes seed
np.random.seed(1)
seed(32611)


# Takes user inputs
num_regions = int(input('Enter number of regions: '))
confidence_style = input('0 for searched confidences, 1 for uniform: ')
loss_func_type = input('0 for weighted error loss, 1 for time loss: ')

initial_bound_type = 0

conf_alpha = 0.05
uniform_confidence = []
for i in range(num_regions):
    uniform_confidence.append(1 - (conf_alpha / num_regions))

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
def split_data(bounds, data):
    full_bounds = [B_LEFT] + bounds + [B_RIGHT]
    trajectories = []
    for i in range(num_regions):
        trajectories.append([])

    for i in range(len(data['RealTrajectories'])):
        real_traj = data['RealTrajectories'][i]
        est_traj = data['EstTrajectories'][i]

        time_weights = []
        for t in range(len(real_traj)):
            time_weights.append(0.9**t)

        cur_real_traj = []
        cur_est_traj = []
        cur_time_weights = []

        indices = np.argsort(real_traj)
        for p in range(len(indices)):
            cur_real_traj.append(real_traj[indices[p]])
            cur_est_traj.append(est_traj[indices[p]])
            cur_time_weights.append(time_weights[indices[p]])

        for j in range(num_regions):
            sub_traj = []
            for k in range(len(cur_real_traj)):
                if cur_real_traj[k] > full_bounds[j] and cur_real_traj[k] <= full_bounds[j+1]:
                    sub_traj.append([cur_real_traj[k], abs(cur_real_traj[k] - cur_est_traj[k]), cur_time_weights[k]])
            if len(sub_traj) > 0:
                trajectories[j].append(sub_traj)
    
    return trajectories

# Returns 99% conformal bound errors for each region
def find_errors(bounds, confs, data):
    trajectories = split_data(bounds, data)

    region_data = []
    errors = []
    region_sizes = []
    time_weights = []

    for i in range(num_regions):
        time_weight = 0
        size = 0
        temp_data = []
        for j in range(len(trajectories[i])):
            cur_max = 0
            for k in range(len(trajectories[i][j])):
                time_weight += trajectories[i][j][k][2]
                if trajectories[i][j][k][1] > cur_max:
                    cur_max = trajectories[i][j][k][1]
                size += 1
            temp_data.append(cur_max)
        temp_data.append(10000000)
        temp_data.sort()
        region_data.append(temp_data)
        region_sizes.append(size)
        time_weights.append(time_weight)

    for i in range(num_regions):
        index = int(np.floor((len(region_data[i])) * confs[i]) - 1)
        if len(region_data[i]) >= 1:
            #print(f'Region data length: {len(region_data[i])}, Index: {index}')
            errors.append(region_data[i][index])
        else:
            errors.append(10000000)

    if loss_func_type == '1':
        return errors, time_weights
    else:        
        return errors, region_sizes

    
# Generates bounds where (x: B1, y: B2) and B1 <= B2
def generate_points(num_bounds):
    bounds_list = []

    for i in range(num_bounds):
        cur_bounds = []
        for j in range(num_regions - 1):
            new_B = np.random.uniform(B_LEFT, B_RIGHT)

            cur_bounds.append(new_B)
        cur_bounds.sort()
        bounds_list.append(cur_bounds)

    return bounds_list

# Calculates the loss function for a given set of bounds
def calculate_loss(bounds, confs, data):
    
    loss = 0
    if loss_func_type == '1':
        errors, time_weights = find_errors(bounds, confs, data)
        for i in range(num_regions):
            loss += errors[i] * time_weights[i]
    else:
        errors, steps = find_errors(bounds, confs, data)
        total_steps = 0
        for i in range(num_regions):
            total_steps += steps[i]

        for i in range(num_regions):
            loss += (steps[i] / total_steps) * errors[i]
    return loss


def generate_initial_bounds(data):
    if initial_bound_type == '1':
        points = generate_points(100)
        losses = []
        count = 0
        # Calculates and saves the loss for each set of bounds
        for point in points:
            print(point)
            losses.append(calculate_loss(point, uniform_confidence))
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

        return point, calculate_loss(point, uniform_confidence, data)

# could explore a bfs-style approach
def simulated_annealiing(point, data):
    s = point
    s_loss = calculate_loss(point, uniform_confidence, data)
    T = 1
    T_min = 0.00001
    alpha = 0.95
    epsilon = 0.0001
    region_change = 0.5
    conf_change = 0.03

    confs = []
    for i in range(num_regions):
        confs.append(1 - uniform_confidence[i])

    while T > T_min:
        for i in range(10):
            s_new = []
            for j in range(len(s)):
                new_B = np.random.uniform(max(s[j] - (region_change * T), B_LEFT), min(s[j] + (region_change * T), B_RIGHT))
                s_new.append(new_B)

            
            confs_new = []
            if confidence_style == '1':
                temp_new_confs = uniform_confidence

                for p in range(num_regions):
                    confs_new.append(1-temp_new_confs[p])
            else:
                conf_sum = 0
                for j in range(num_regions):
                    if j < (num_regions - 1):
                        lower = max(confs[j] - (conf_change * T), 0)
                        upper = min(confs[j] + (conf_change * T), conf_alpha - conf_sum - (num_regions - (j+1)) * (epsilon))
                        #assert lower <= upper, "doesnt work"


                        confs_new.append(np.random.uniform(lower, upper))
                    else:
                        confs_new.append(conf_alpha - conf_sum)

                    conf_sum += confs_new[j]
            
                
                temp_new_confs = []
                for p in range(num_regions):
                    temp_new_confs.append(1-confs_new[p])
            

            s_new.sort()
            s_new_loss = calculate_loss(s_new, temp_new_confs, data)

            if s_new_loss < s_loss:
                #print(f'New loss: {s_new_loss}')
                #print(f'Old loss: {s_loss}')
                s = s_new
                s_loss = s_new_loss
                confs = confs_new
        
        T = T * alpha
        print(f'{T}, {T_min}')
    
    returned_confs = []
    for i in range(num_regions):
        returned_confs.append(1 - confs[i])

    return s, returned_confs

# Need to keep same structure below to ensure same results
# Splits data into two datasets
total_size = len(data['RealTrajectories'])
data_one = reduce_data(data, 0.5, total_size)
data_two = reduce_data(data, 0.5, total_size)
# alpha_data is for region tuning, bounds_data for finding errors
data_one_size = len(data_one['RealTrajectories'])
alpha_data = reduce_data(data_one, 0.25, data_one_size)
bounds_data = reduce_data(data_one, 0.75, data_one_size)


initial_bounds, initial_loss = generate_initial_bounds(alpha_data)

# Generates n sets of bounds
print(f'{initial_loss}, {initial_bounds}')
print(calculate_loss(initial_bounds, uniform_confidence, alpha_data))

# Displays initial data, pre simulated annealing
plt.title('Perception Error vs Position')
plt.xlabel('Position')
plt.ylabel('Perception Error')
plt.scatter(pos_data, error_data, s=1)
plt.axvline(x = B_LEFT, color = 'black', label = 'Edge')
plt.axvline(x = B_RIGHT, color = 'black', label = 'Edge')
for i in range(len(initial_bounds)):
    plt.axvline(x = initial_bounds[i], color = 'b')    
plt.legend(loc = 1, prop={'size': 6})
plt.show()

# Uses simulated annealing to find more optimal set of bounds using the current set
new_bounds, new_confidences = simulated_annealiing(initial_bounds, bounds_data)
new_loss = calculate_loss(new_bounds, new_confidences, alpha_data)

old_errors = find_errors(initial_bounds, uniform_confidence, alpha_data)[0]
new_errors = find_errors(new_bounds, new_confidences, alpha_data)[0]

print('--- Pre simulated annealing: ---')
print(f'Loss: {initial_loss}\nBounds: {initial_bounds}\n99% Errors: {old_errors}')
print('--- Post simulated annealing: ---')
print(f'Loss: {new_loss}')


for i in range(num_regions):
    if i == 0:
        print(f'[{round(B_LEFT, 6)}, {round(new_bounds[i], 6)}]: {round(new_errors[i], 6)}')
    elif i == num_regions - 1:
        print(f'[{round(new_bounds[i - 1], 6)}, {round(B_RIGHT, 6)}]: {round(new_errors[i], 6)}')
    else:
        print(f'[{round(new_bounds[i - 1], 6)}, {round(new_bounds[i], 6)}]: {round(new_errors[i], 6)}')

print(f'Confidences: {new_confidences}')

# Displays new data, post simulated annealing
plt.title('Perception Error vs Position')
plt.xlabel('Position')
plt.ylabel('Perception Error')
plt.scatter(pos_data, error_data, s=1)
plt.axvline(x = B_LEFT, color = 'black', label = 'Edge')
plt.axvline(x = B_RIGHT, color = 'black', label = 'Edge')
for i in range(len(initial_bounds)):
    plt.axvline(x = initial_bounds[i], color = 'b')    
for i in range(len(new_bounds)):
    plt.axvline(x = new_bounds[i], color = 'r')    
plt.legend(loc = 1, prop={'size': 6})
plt.show()


region_errors = find_errors(new_bounds, new_confidences, alpha_data)[0]
print("Rounded Values:")
for i in range(num_regions):
    if i == 0:
        print(f'[{round(B_LEFT, 6)}, {round(new_bounds[i], 6)}]: {round(region_errors[i], 6)}')
    elif i == num_regions - 1:
        print(f'[{round(new_bounds[i - 1], 6)}, {round(B_RIGHT, 6)}]: {round(region_errors[i], 6)}')
    else:
        print(f'[{round(new_bounds[i - 1], 6)}, {round(new_bounds[i], 6)}]: {round(region_errors[i], 6)}')

print("Unrounded Values:")
for i in range(num_regions):
    if i == 0:
        print(f'[{B_LEFT}, {new_bounds[i]}]: {region_errors[i]}')
    elif i == num_regions - 1:
        print(f'[{new_bounds[i - 1]}, {B_RIGHT}]: {region_errors[i]}')
    else:
        print(f'[{new_bounds[i - 1]}, {new_bounds[i]}]: {region_errors[i]}')