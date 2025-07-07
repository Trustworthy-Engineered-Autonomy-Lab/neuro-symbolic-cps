import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

def draw_trajectories_with_split(data_filename, best_split):
    # Load dataset
    with open(data_filename, 'rb') as f:
        all_trajectory_dict = pickle.load(f)

    all_state_trajectories = all_trajectory_dict['RealStates']
    x_y_trajectories = [traj[:, :2] for traj in all_state_trajectories]

    # Compute bounding box
    all_points = np.vstack(x_y_trajectories)
    multi_point = MultiPoint(all_points)
    bounding_box = multi_point.envelope
    minx, miny, maxx, maxy = bounding_box.bounds

    # Unpack best split
    x_split, y_split = best_split

    # Plot all trajectories
    plt.figure(figsize=(10, 10))
    for traj in x_y_trajectories:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.1, color='gray')

    # Draw split lines only
    plt.axvline(x=x_split, color='blue', linestyle='--', linewidth=2, label='x_split')
    plt.axhline(y=y_split, color='green', linestyle='--', linewidth=2, label='y_split')

    plt.title("Trajectories with Optimized x/y Split")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


data_filename = "./f110_dataset_06_26_2025.pkl"
best_split = [-0.2196722816635, 6.602411944981617]
draw_trajectories_with_split(data_filename, best_split)
