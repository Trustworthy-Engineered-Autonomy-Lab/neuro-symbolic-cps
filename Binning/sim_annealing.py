import pickle
import matplotlib.pyplot as plt
import numpy as np

# Loads position and error data (x: position, y: error)
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
except:
    exit(1)

sorted_data = pos_data.copy()
sorted_data.sort()
B0 = sorted_data[0]
B3 = sorted_data[len(sorted_data) - 1]
print(f'{B0}, {B3}')

# Displays initial data
plt.title('Perception Error vs Position')
plt.xlabel('Position')
plt.ylabel('Perception Error')
plt.scatter(pos_data, error_data, s=1)
plt.show()

# Constants for left and right bounds (B0 < B1 < B2 < B3)
#B0 = -1.2
#B3 = 0.6

# Finds 95% error for each region
def find_errors(B1, B2):
    bounds = [B0, B1, B2, B3]
    errors = [0, 0, 0]

    # Split data into three regions
    region_data = [[],[],[]]
    for i in range(3):
        temp_data = []
        for j in range(len(pos_data)):
            if pos_data[j] <= bounds[i + 1] and pos_data[j] >= bounds[i]:
                temp_data.append(abs(error_data[j]))
        temp_data.append(100000)
        temp_data.sort()
        region_data[i] = temp_data

    # Finds 99% error for each region
    for i in range(3):
        index = int(np.ceil((len(region_data[i])) * 0.99) - 1)
        if len(region_data[i]) >= 1:
            errors[i] = region_data[i][index]
        else:
            errors[i] = 0

    return errors

# Generates points where (x: B1, y: B2) and B1 <= B2
def generate_points(num_points):
    points = []

    for i in range(num_points):
        # Calculates points such that B0 < B1 < B2 < B3
        new_B1 = np.random.uniform(B0, B3 - 0.01)
        new_B2 = np.random.uniform(new_B1, B3)

        points.append((new_B1, new_B2))

    return points

# Calculates the loss function for a given set of bounds
def calculate_loss(point):
    errors = find_errors(point[0], point[1])
    loss = ((point[0] - B0) * errors[0]) + ((point[1] - point[0]) * errors[1]) + ((B3 - point[1]) * errors[2])
    
    return loss


# could explore a bfs-style approach
def simulated_annealiing(point):
    s = point
    s_loss = calculate_loss(point)
    T = 1
    T_min = 0.0001
    alpha = 0.9

    while T > T_min:
        for i in range(10):
            B1_new = np.random.uniform(max(s[0] - (0.5 * T), B0), min(s[1] + (0.5 * T), B3 - 0.01))
            B2_new = np.random.uniform(max(s[1] - (0.5 * T), B1_new), min(s[1] + (0.5 * T), B3))
            s_new = (B1_new, B2_new)
        
            s_new_loss = calculate_loss(s_new)
            if s_new_loss < s_loss:
                s = s_new
                s_loss = s_new_loss
        
        T = T * alpha
        print(f'{T}, {T_min}')
    
    return s

# Generates n sets of bounds
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

print(f'{min_loss}, {points[min_index]}')

# Displays initial data, pre simulated annealing
plt.title('Perception Error vs Position')
plt.xlabel('Position')
plt.ylabel('Perception Error')
plt.scatter(pos_data, error_data, s=1)
plt.axvline(x = points[min_index][0], color = 'b', label = 'B1')
plt.axvline(x = points[min_index][1], color = 'g', label = 'B2')
plt.legend(loc = 1, prop={'size': 6})
plt.show()

# Uses simulated annealing to find more optimal set of bounds using the current set
new_min_point = simulated_annealiing(points[min_index])
new_loss = calculate_loss(new_min_point)

print('Pre simulated annealing:')
print(f'Loss: {min_loss}, Bounds: {points[min_index]}, 99% Errors: {find_errors(points[min_index][0], points[min_index][1])}')
print('Post simulated annealing')
print(f'Loss: {new_loss}, Bounds: {new_min_point}, 99% Errors: {find_errors(new_min_point[0], new_min_point[1])}')

# Displays new data, post simulated annealing
plt.title('Perception Error vs Position')
plt.xlabel('Position')
plt.ylabel('Perception Error')
plt.scatter(pos_data, error_data, s=1)
plt.axvline(x = points[min_index][0], color = 'b', label = 'Initial B1')
plt.axvline(x = points[min_index][1], color = 'g', label = 'Initial B2')
plt.axvline(x = new_min_point[0], color = 'r', label = 'New B1')
plt.axvline(x = new_min_point[1], color = 'y', label = 'New B2')
plt.legend(loc = 1, prop={'size': 6})
plt.show()
