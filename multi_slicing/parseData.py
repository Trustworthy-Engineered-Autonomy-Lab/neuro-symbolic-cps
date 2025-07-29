import matplotlib.pyplot as plt
import gzip
import math
import random
import pickle

import numpy as np

def main():
    data_filename = "./f110_dataset_06_26_2025.pkl"
    
    with open(data_filename, 'rb') as f:
        all_trajectory_dict = pickle.load(f)
    for key in all_trajectory_dict.keys():
        print(key)

    fig = plt.figure()
    plt.ylim((-0.001,0.2))
    for i in range(len(all_trajectory_dict['1DErrors'])):
        plt.plot(all_trajectory_dict['1DErrors'][i], 'k.')
    plt.savefig("./Absolute Perception errors vs Time.png")
    plt.close()
    
    all_state_trajectories = all_trajectory_dict['RealStates']
    all_perception_errors = all_trajectory_dict['1DErrors']
    
    # example trajectory
    # index 0: x pos in [-0.75, 5], starts in [-0.1, 0.1]
    # index 1: y pos in [3.5, 10], starts in [3.5, 4.5]
    # index 2: theta in [-pi, pi], starts at pi/2 and ends close to 0
    # index 3: velocity in [0, 3ish], starts and 0 and maxs out before 2.5 m/s

    print("First trajectory, first timestep, x: {}, y: {}, theta: {}, vel: {}\n perception error: {}".format(
        all_state_trajectories[0][0][0],
        all_state_trajectories[0][0][1],
        all_state_trajectories[0][0][2],
        all_state_trajectories[0][0][3],
        all_perception_errors[0][0],
    ))
    
    # take half for conformal prediction, half for testing (repeated from last time)
    random.seed(0)
    conf_fraction = 0.5
    conf_traj_inds = random.sample(range(len(all_state_trajectories)), int(conf_fraction*len(all_state_trajectories)))
    test_traj_inds = list(set(range(len(all_state_trajectories)))-set(conf_traj_inds))

    conf_state_trajectories = [all_state_trajectories[i] for i in conf_traj_inds]
    conf_perception_errors = [all_perception_errors[i] for i in conf_traj_inds]
    
    test_state_trajectories = [all_state_trajectories[i] for i in test_traj_inds]
    test_perception_errors = [all_perception_errors[i] for i in test_traj_inds]

    ## data extracted, now get regional conformal bounds
    print("Total Trajectories in dataset: {}".format(len(all_state_trajectories)))
    print("Number of Trajectories for deriving conformal bounds: {}".format(len(conf_state_trajectories)))
    print("Number of Trajectories for testing bounds: {}".format(len(test_state_trajectories)))

if __name__ == '__main__':
    main()