import numpy as np
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time
import pickle
import random
import math
import matplotlib.pyplot as plt
from gurobipyTutorial import optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBound, \
    optimzeTimeAlphasKKTNoMaxLowerBoundMinArea

import random
from random import sample, seed

# Initializes seed
seed(32611)

PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100


def computeRValuesIndiv(x_vals, y_vals, x_hats, y_hats):
    R_vals = [[math.sqrt((x_vals[i][j] - x_hats[i][j]) ** 2 + (y_vals[i][j] - y_hats[i][j]) ** 2) for j in
               range(len(x_vals[i]))] for i in range(len(x_vals))]

    return R_vals


def computeCPCirlce(x_vals, y_vals, x_hats, y_hats, delta):
    R_vals = [math.sqrt((x_vals[i] - x_hats[i]) ** 2 + (y_vals[i] - y_hats[i]) ** 2) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return (R_vals[ind_to_ret])


def computeCPFixedAlphas(x_vals, y_vals, x_hats, y_hats, alphas, delta):
    R_vals = [
        max([alphas[j] * math.sqrt((x_vals[i][j] - x_hats[i][j]) ** 2 + (y_vals[i][j] - y_hats[i][j]) ** 2) for j in
             range(len(x_vals[i]))]) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]


def computeCPFixedAlphas_MC(Rs, alphas, delta):
    R_vals = [max([alphas[j] * Rs[i][j] for j in range(len(Rs[i]))]) for i in range(len(Rs))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]


def computeCoverageRAndAlphas_MC(Rs, alphas, D_cp):
    R_vals = [max([alphas[j] * Rs[i][j] for j in range(len(Rs[i]))]) for i in range(len(Rs))]
    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within) / len(R_vals)
    return coverage_pct


def computeCoverageRAndAlphas_MC_yuang(Rs, D_cp):
    R_vals = [r for sublist in Rs for r in sublist]  # Flatten the list of lists
    # R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]
    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within) / len(R_vals)
    return coverage_pct


def computeCoverageRAndAlphas(x_vals, y_vals, x_hats, y_hats, alphas, D_cp):
    R_vals = [
        max([alphas[j] * math.sqrt((x_vals[i][j] - x_hats[i][j]) ** 2 + (y_vals[i][j] - y_hats[i][j]) ** 2) for j in
             range(len(x_vals[i]))]) for i in range(len(x_vals))]

    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within) / len(R_vals)
    return coverage_pct


def computeCoverageCircle(x_vals, y_vals, x_hats, y_hats, Ds_cp):
    coverage_count = 0
    for i in range(len(x_vals)):

        temp = sum(
            [1 if math.sqrt((x_vals[i][j] - x_hats[i][j]) ** 2 + (y_vals[i][j] - y_hats[i][j]) ** 2) > Ds_cp[j] else 0
             for j in range(len(x_vals[i]))])
        if temp == 0:
            coverage_count += 1

    coverage_pct = float(coverage_count) / len(x_vals)
    return coverage_pct


def plot_circle(x, y, size, color="-b", label=None):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]

    if label is None:
        plt.plot(xl, yl, color)
    else:
        plt.plot(xl, yl, color, label=label)


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


##### This is the code for the position as the non-conformity scores #####
if __name__ == "__main__":

    filename = 'data/TrajData4000_newinit.pkl'
    with open(filename, 'rb') as f:
        all_data = pickle.load(f)

    Est_trajectory = all_data['EstTrajectories']
    perception_error = all_data['Errors']


    # Ensure `data` is defined before using it
    total_size = len(all_data['Errors'])
    # First reduction: Split data into two parts
    data_one = reduce_data(all_data, 0.5, total_size)
    data_two = reduce_data(all_data, 0.5, total_size)


### Find the max and min for the trajectory
    ## Find the maximum and minimal trajectory in the remaining.
    new_traj = data_one['EstTrajectories']
    p_len = 90
    new_traj = [
        sublist[:p_len] for sublist in new_traj
    ]
    new_traj = np.array(new_traj)

    max_trajectory = np.max(new_traj, axis=0)
    min_trajectory = np.min(new_traj, axis=0)

    # np.random.seed(42)  # For reproducibility
    # new_traj1 = np.random.rand(2000, 90)  # Generating random trajectories
    # # Finding the max and min for each step across all trajectories
    # max_values = np.max(new_traj1, axis=0)  # Max value for each of the 90 steps
    # min_values = np.min(new_traj1, axis=0)  # Min value for each of the 90 steps

    max_trajectory = np.array(max_trajectory)  # Shape: (90, state_dim) expected
    min_trajectory = np.array(min_trajectory)  # Shape: (90, state_dim) expected
    # Debugging: Print actual shapes before trimming
    print("Original Max Trajectory Shape:", max_trajectory.shape)
    print("Original Min Trajectory Shape:", min_trajectory.shape)

    # Ensure both trajectories have exactly 90 time steps
    time_steps = 90
    min_trajectory = min_trajectory[:time_steps]  # Trim to first 90 steps
    max_trajectory = max_trajectory[:time_steps]  # Trim to first 90 steps



    # Further split data_one
    data_one_size = len(data_two['Errors'])  # Fixed missing closing bracket
    alpha_data = reduce_data(data_two, 0.025, data_one_size)
    bounds_data = reduce_data(data_two, 0.975, data_one_size)

    alpha_disp = alpha_data['Errors']
    cp_caculate_disp = bounds_data['Errors']


##### Define the dataset ######
    p_len = 90
    delta = 0.05 ## Make upper and lower bounds 0.975
    R_vals_calib_alpha_MC = alpha_disp
    R_vals_calib_alpha_MC = [
        sublist[:p_len] for sublist in R_vals_calib_alpha_MC
    ]

    calibrate_R = cp_caculate_disp
    calibrate_R = [
        sublist[:p_len] for sublist in calibrate_R
    ]

    # D_cp = computeCPFixedAlphas_MC(calibrate_R, alphas, delta)
    # print("The final confromal bounds: ", D_cp)

    ###### run optimziation########
    M = 100000
    start_time = time.time()
    m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals_calib_alpha_MC, delta, M)
    end_time = time.time()

    start_time_milp = time.time()
    m_milp = optimzeTimeAlphasKKT(R_vals_calib_alpha_MC, delta, M)
    end_time_milp = time.time()

    print("Solve time: " + str(end_time - start_time))
    print("Solve time MILP: " + str(end_time_milp - start_time_milp))
    # print("Solve time min area: " + str(end_time_min_area-start_time_min_area))

    alphas = []
    for v in m.getVars():
        if "alphas" in v.varName:
            alphas.append(v.x)
        if "q" in v.varName:
            # print(v.x)
            print("obj: " + str(v.x))

    alphas_milp = []
    for v in m_milp.getVars():
        if "alphas" in v.varName:
            alphas_milp.append(v.x)
        if "q" in v.varName:
            # print(v.x)
            print("obj MILP: " + str(v.x))

    print("alphas: " + str(alphas))
    print("alphas MILP: " + str(alphas_milp))

    plt.plot(alphas, 'k*')
    plt.xlabel("Prediction Horizon", fontsize="16")
    plt.ylabel("Alpha Value", fontsize="16")

    plt.savefig('images/forPaper/alphas_MC.png')

    ## run CP using alpha values on remaining calibration data
    calibrate_R = cp_caculate_disp
    calibrate_R = [
        sublist[:p_len] for sublist in calibrate_R
    ]
    D_cp = computeCPFixedAlphas_MC(calibrate_R, alphas, delta)
    print("The final confromal bounds: ", D_cp)
## Solve the perception error bounds.
    cp_perception_step = [D_cp / value for value in alphas]

## Add it to the top
inflate_max_trajectory = max_trajectory + cp_perception_step
inflate_min_trajectory = min_trajectory - cp_perception_step

# Plot each state dimension separately
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 5))
plt.plot(range(time_steps), max_trajectory, label="Max Trajectory", linestyle="dashed", color="blue")
plt.plot(range(time_steps), min_trajectory, label="Min Trajectory", linestyle="dashed", color="blue")
plt.plot(range(time_steps), inflate_max_trajectory, label="Inflate upper Trajectory", linestyle="solid", color="red")
plt.plot(range(time_steps), inflate_min_trajectory, label="Inflate lower Trajectory", linestyle="solid", color="red")

# Labels and title
plt.xlabel("Time Steps")
plt.ylabel("State Values")
plt.title("Max and Min Trajectories Over 90 Time Steps")
plt.legend()
plt.grid()

# Show the plot
plt.show()

print("stop here")

######### This is the code for the position as the non-conformity scores #########
# if __name__ == "__main__":
#
#     try:
#         filename = 'data/data.pkl'
#         with open(filename, 'rb') as f:
#             data = pickle.load(f)
#     except:
#         exit(1)
#
#     try:
#         filename = 'data/pos_data.pkl'
#         with open(filename, 'rb') as f:
#             pos_data = pickle.load(f)
#     except:
#         exit(1)
#     try:
#         filename = 'data/error_data.pkl'
#         with open(filename, 'rb') as f:
#             error_data = pickle.load(f)
#         error_data = [abs(error) for error in error_data]
#     except:
#         exit(1)
#
#     sorted_data = pos_data.copy()
#     sorted_data.sort()
#     B_LEFT = sorted_data[0]
#     B_RIGHT = sorted_data[len(sorted_data) - 1]
#     print(f'{B_LEFT}, {B_RIGHT}')
#
#     # Ensure `data` is defined before using it
#     total_size = len(data['RealTrajectories'])
#
#     # First reduction: Split data into two parts
#     data_one = reduce_data(data, 0.5, total_size)
#     data_two = reduce_data(data, 0.5, total_size)
#
#     # Further split data_one
#     data_one_size = len(data_one['RealTrajectories'])  # Fixed missing closing bracket
#     alpha_data = reduce_data(data_two, 0.05, data_one_size)
#     bounds_data = reduce_data(data_two, 0.95, data_one_size)
#
#     # alpha_disp = alpha_data['Errors']
#     # cp_caculate_disp = bounds_data['Errors']
#
#     position_time_step = alpha_data['EstTrajectories']
#     cp_caculate_position = bounds_data['EstTrajectories']
#
# ##### Define the dataset ######
#     p_len = 90
#     delta = 0.2 ## Make upper and lower bounds 0.975
#     R_vals_calib_alpha_MC = position_time_step
#     R_vals_calib_alpha_MC = [
#         sublist[:p_len] for sublist in R_vals_calib_alpha_MC
#     ]
#
#     calibrate_R = cp_caculate_position
#     calibrate_R = [
#         sublist[:p_len] for sublist in calibrate_R
#     ]
#     # D_cp = computeCPFixedAlphas_MC(calibrate_R, alphas, delta)
#     # print("The final confromal bounds: ", D_cp)
#
#     ###### run optimziation########
#     M = 100000
#     start_time = time.time()
#     m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals_calib_alpha_MC, delta, M)
#     end_time = time.time()
#
#     start_time_milp = time.time()
#     m_milp = optimzeTimeAlphasKKT(R_vals_calib_alpha_MC, delta, M)
#     end_time_milp = time.time()
#
#     print("Solve time: " + str(end_time - start_time))
#     print("Solve time MILP: " + str(end_time_milp - start_time_milp))
#     # print("Solve time min area: " + str(end_time_min_area-start_time_min_area))
#
#     alphas = []
#     for v in m.getVars():
#         if "alphas" in v.varName:
#             alphas.append(v.x)
#         if "q" in v.varName:
#             # print(v.x)
#             print("obj: " + str(v.x))
#
#     alphas_milp = []
#     for v in m_milp.getVars():
#         if "alphas" in v.varName:
#             alphas_milp.append(v.x)
#         if "q" in v.varName:
#             # print(v.x)
#             print("obj MILP: " + str(v.x))
#
#     print("alphas: " + str(alphas))
#     print("alphas MILP: " + str(alphas_milp))
#
#     plt.plot(alphas, 'k*')
#     plt.xlabel("Prediction Horizon", fontsize="16")
#     plt.ylabel("Alpha Value", fontsize="16")
#
#     plt.savefig('images/forPaper/alphas_MC.png')
#
#     ## run CP using alpha values on remaining calibration data
#     calibrate_R = cp_caculate_position
#     calibrate_R = [
#         sublist[:p_len] for sublist in calibrate_R
#     ]
#     D_cp = computeCPFixedAlphas_MC(calibrate_R, alphas, delta)
#     print("The final confromal bounds: ", D_cp)