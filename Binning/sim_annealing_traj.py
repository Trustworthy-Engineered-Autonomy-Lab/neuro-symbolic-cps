import pickle
import matplotlib.pyplot as plt
import numpy as np
from random import sample, seed

# Initializes seed
seed(32611)

# Takes user inputs
#data_percent = float(input('Enter percent of data to use (0.0,1.0): '))
num_regions = int(input('Enter number of regions: '))
initial_bound_type = input('0 for uniform, 1 for semi-random: ')

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
        cur_real_traj = data['RealTrajectories'][i]
        cur_est_traj = data['EstTrajectories'][i]
        cur_real_traj.sort()
        cur_est_traj.sort()

        for j in range(num_regions):
            sub_traj = []
            for k in range(len(cur_real_traj)):
                if cur_real_traj[k] > full_bounds[j] and cur_real_traj[k] <= full_bounds[j+1]:
                    sub_traj.append((cur_real_traj[k], abs(cur_real_traj[k] - cur_est_traj[k])))
            trajectories[j].append(sub_traj)
    
    return trajectories
        

# Returns 99% conformal bound errors for each region
def find_errors(bounds, data):
    trajectories = split_data(bounds, data)

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
        index = int(np.ceil((len(region_data[i])) * (0.99 + (num_regions - 1)/(num_regions * 100))) - 1)
        if len(region_data[i]) >= 1:
            errors.append(region_data[i][index])
        else:
            errors.append(100000)
    
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
def calculate_loss(bounds, data):
    errors, steps = find_errors(bounds, data)

    # Previous loss function that uses width of bins
    #loss = ((point[0] - B0) * errors[0]) + ((point[1] - point[0]) * errors[1]) + ((B3 - point[1]) * errors[2])
    
    total_steps = 0
    for i in range(num_regions):
        total_steps += steps[i]

    loss = 0
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

# Need to keep same structure below to ensure same results
# Splits data into two datasets
total_size = len(data['RealTrajectories'])
data_one = reduce_data(data, 0.5, total_size)
data_two = reduce_data(data, 0.5, total_size)
# alpha_data is for region tuning, bounds_data for finding errors
data_one_size = len(data_one['RealTrajectories'])
alpha_data = reduce_data(data_one, 0.4, data_one_size)
bounds_data = reduce_data(data_one, 0.6, data_one_size)
# Example implementation for method 2:
# data_two_size = len(data_two['RealTrajectories'])
# ____ = reduce(data_two, percent, data_two_size)
# ____ = reduce(data_two, percent, data_two_size)

initial_bounds, initial_loss = generate_initial_bounds(alpha_data)

# Generates n sets of bounds
print(f'{initial_loss}, {initial_bounds}')
print(calculate_loss(initial_bounds, alpha_data))

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
new_bounds = simulated_annealiing(initial_bounds, alpha_data)
new_loss = calculate_loss(new_bounds, alpha_data)

old_errors = find_errors(initial_bounds, alpha_data)[0]
new_errors = find_errors(new_bounds, alpha_data)[0]

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
