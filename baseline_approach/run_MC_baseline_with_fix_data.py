import numpy as np
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time
import pickle
import random
import math
import matplotlib.pyplot as plt
from gurobipyTutorial import optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKTNoMaxLowerBoundMinArea


import random
from random import sample, seed
# Initializes seed
seed(32611)

PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100

def computeRValuesIndiv(x_vals,y_vals,x_hats,y_hats):

    R_vals = [ [math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))]  for i in range(len(x_vals))  ]

    return R_vals



def computeCPCirlce(x_vals,y_vals,x_hats,y_hats,delta):

    R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])



def computeCPFixedAlphas(x_vals,y_vals,x_hats,y_hats,alphas,delta):

    R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return R_vals[ind_to_ret]


def computeCPFixedAlphas_MC(Rs, alphas, delta):
    R_vals = [max([alphas[j] * Rs[i][j] for j in range(len(Rs[i]))]) for i in range(len(Rs))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals) * (1 - delta))
    return R_vals[ind_to_ret]

def computeCoverageRAndAlphas_MC(Rs,alphas,D_cp):
    R_vals = [max([alphas[j] * Rs[i][j] for j in range(len(Rs[i]))]) for i in range(len(Rs))]
    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct

def computeCoverageRAndAlphas_MC_yuang(Rs,D_cp):
    R_vals = [r for sublist in Rs for r in sublist]  # Flatten the list of lists
    #R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]
    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct

def computeCoverageRAndAlphas(x_vals,y_vals,x_hats,y_hats,alphas,D_cp):

    R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]

    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct


def computeCoverageCircle(x_vals,y_vals,x_hats,y_hats,Ds_cp):
    coverage_count = 0
    for i in range(len(x_vals)):

        temp = sum([1 if math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) > Ds_cp[j] else 0 for j in range(len(x_vals[i]))])
        if temp == 0:
            coverage_count += 1
        

    coverage_pct = float(coverage_count)/len(x_vals)
    return coverage_pct


def plot_circle(x, y, size, color="-b", label=None):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]

    if label is None:
        plt.plot(xl, yl, color)
    else:
        plt.plot(xl, yl, color,label=label)


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


if __name__ == "__main__":

    try:
        filename = 'data/data.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except:
        exit(1)

    try:
        filename = 'data/pos_data.pkl'
        with open(filename, 'rb') as f:
            pos_data = pickle.load(f)
    except:
        exit(1)
    try:
        filename = 'data/error_data.pkl'
        with open(filename, 'rb') as f:
            error_data = pickle.load(f)
        error_data = [abs(error) for error in error_data]
    except:
        exit(1)

    sorted_data = pos_data.copy()
    sorted_data.sort()
    B_LEFT = sorted_data[0]
    B_RIGHT = sorted_data[len(sorted_data) - 1]
    print(f'{B_LEFT}, {B_RIGHT}')

    # Ensure `data` is defined before using it
    total_size = len(data['RealTrajectories'])

    # First reduction: Split data into two parts
    data_one = reduce_data(data, 0.5, total_size)
    data_two = reduce_data(data, 0.5, total_size)

    # Further split data_one
    data_one_size = len(data_one['RealTrajectories'])  # Fixed missing closing bracket
    alpha_data = reduce_data(data_two, 0.025, data_one_size)
    bounds_data = reduce_data(data_two, 0.975, data_one_size)

    alpha_disp = alpha_data['Errors']
    cp_caculate_disp = bounds_data['Errors']


    p_len = 90
    delta = 0.05
    R_vals_calib_alpha_MC = alpha_disp
    R_vals_calib_alpha_MC = [
        sublist[:p_len] for sublist in R_vals_calib_alpha_MC
    ]

    # alphas = 
    calibrate_R = cp_caculate_disp
    calibrate_R = [
        sublist[:p_len] for sublist in calibrate_R
    ]
    # D_cp = computeCPFixedAlphas_MC(calibrate_R,alphas,delta)
    # print("The final confromal bounds: ", D_cp)

    ## run optimziation
    #M = 100000  # big value for linearization of max constraint
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
    D_cp = computeCPFixedAlphas_MC(calibrate_R,alphas,delta)
    print("The final confromal bounds: ", D_cp)

    # validate_R = discrepancy_data[700:1400]
    # validate_R = [
    #     sublist[:p_len] for sublist in validate_R
    # ]
    # validation_coverage_MC = computeCoverageRAndAlphas_MC(validate_R,alphas,D_cp)
    # print("Their validation coverage MC: " + str(validation_coverage_MC))
    #
    #
    # validation_coverage_MC1 = computeCoverageRAndAlphas_MC_yuang(validate_R,D_cp)
    # print("Real validation coverage MC: " + str(validation_coverage_MC1))




    ###########################
    discrepancy_data = data['Errors']
    p_len = 90
    delta = 0.01
    trace_for_alphas = 50 ## 100 trajectories for alphas
    makePlots = True
    random.seed(34956)

    #1. We get 100 trajectories to calculate the R values
    # R_vals_calib_alpha = computeRValuesIndiv(all_x_calib_alphas, all_y_calib_alphas, all_x_l_calib_alphas,
    #                                          all_y_l_calib_alphas)
    R_vals_calib_alpha_MC = discrepancy_data[0:trace_for_alphas]
    R_vals_calib_alpha_MC = [
        sublist[:p_len] for sublist in R_vals_calib_alpha_MC
    ]
    ## run optimziation
    M = 100000  # big value for linearization of max constraint

    start_time = time.time()
    m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals_calib_alpha_MC, delta, M)
    end_time = time.time()

    start_time_milp = time.time()
    m_milp = optimzeTimeAlphasKKT(R_vals_calib_alpha_MC, delta, M)
    end_time_milp = time.time()

    # start_time_min_area = time.time()
    # m_min_area = optimzeTimeAlphasKKTNoMaxLowerBoundMinArea(R_vals_calib_alpha,delta,M)
    # end_time_min_area = time.time()

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
    calibrate_R = discrepancy_data[trace_for_alphas:700]
    calibrate_R = [
        sublist[:p_len] for sublist in calibrate_R
    ]
    D_cp = computeCPFixedAlphas_MC(calibrate_R,alphas,delta)

    validate_R = discrepancy_data[700:1400]
    validate_R = [
        sublist[:p_len] for sublist in validate_R
    ]
    validation_coverage_MC = computeCoverageRAndAlphas_MC(validate_R,alphas,D_cp)
    print("Their validation coverage MC: " + str(validation_coverage_MC))


    validation_coverage_MC1 = computeCoverageRAndAlphas_MC_yuang(validate_R,D_cp)
    print("Real validation coverage MC: " + str(validation_coverage_MC1))

