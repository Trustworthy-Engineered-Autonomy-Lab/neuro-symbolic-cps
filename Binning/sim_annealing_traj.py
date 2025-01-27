import pickle
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

data_percent = float(input('Enter percent of data to use (0.0,1.0): '))
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
    error_data = [abs(error) for error in error_data]
except:
    exit(1)

# Finds edge bounds

sorted_data = pos_data.copy()
sorted_data.sort()
B_LEFT = sorted_data[0]
B_RIGHT = sorted_data[len(sorted_data) - 1]
print(f'{B_LEFT}, {B_RIGHT}')


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
        

"""
def split_data(bounds):
    full_bounds = [B_LEFT] + bounds + [B_RIGHT]
    trajectories = []
    for i in range(num_regions):
        trajectories.append([])

    for i in range(len(data['RealTrajectories'])):
        flag = -1
        for j in range(len(data['RealTrajectories'][i])):
            if flag == -1:
                if j == 0:
                    temp_traj = [(data['RealTrajectories'][i][j], data['Errors'][i][j])]

                for k in range(num_regions):
                    if temp_traj[0][0] < full_bounds[k + 1]:
                        flag = k + 1
                        break

            else:
                if data['RealTrajectories'][i][j] <= full_bounds[flag] and data['RealTrajectories'][i][j] >= full_bounds[flag - 1]:
                    temp_traj.append((data['RealTrajectories'][i][j], data['Errors'][i][j]))
                    if j == len(data['RealTrajectories'][i]) - 1:
                        trajectories[flag - 1].append(temp_traj)
                else:
                    trajectories[flag - 1].append(temp_traj)
                    flag = -1
                    temp_traj = [(data['RealTrajectories'][i][j], data['Errors'][i][j])]

    return trajectories
"""

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
def calculate_loss(bounds):
    errors, steps = find_errors(bounds)

    # Previous loss function that uses width of bins
    #loss = ((point[0] - B0) * errors[0]) + ((point[1] - point[0]) * errors[1]) + ((B3 - point[1]) * errors[2])
    
    total_steps = 0
    for i in range(num_regions):
        total_steps += steps[i]

    loss = 0
    for i in range(num_regions):
        loss += (steps[i] / total_steps) * errors[i]

    return loss


def generate_initial_bounds():
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

        return point, calculate_loss(point)

# could explore a bfs-style approach
def simulated_annealiing(point):
    s = point
    s_loss = calculate_loss(point)
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
            s_new_loss = calculate_loss(s_new)

            if s_new_loss < s_loss:
                s = s_new
                s_loss = s_new_loss
        
        T = T * alpha
        print(f'{T}, {T_min}')
    
    return s


reduce_data(data_percent)

initial_bounds, initial_loss = generate_initial_bounds()

# Generates n sets of bounds
print(f'{initial_loss}, {initial_bounds}')

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
new_bounds = simulated_annealiing(initial_bounds)
new_loss = calculate_loss(new_bounds)

print('Pre simulated annealing:')
print(f'Loss: {initial_loss}, Bounds: {initial_bounds}, 99% Errors: {find_errors(initial_bounds)[0]}')
print('Post simulated annealing:')
print(f'Loss: {new_loss}, Bounds: {new_bounds}, 99% Errors: {find_errors(new_bounds)[0]}')


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
