import random
from shapely.geometry import Polygon, LineString, MultiPoint
import matplotlib.pyplot as plt
import gzip
import math
import pickle
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import split as split_geom
import numpy as np

# Fitness function
def fitness_function(x_split, y_split, bounds, x_y_trajectories, all_perception_errors):
    minx, miny, maxx, maxy = bounds

    # Define subregions
    subregions = [
        Polygon([(minx, miny), (x_split, miny), (x_split, y_split), (minx, y_split)]),
        Polygon([(x_split, miny), (maxx, miny), (maxx, y_split), (x_split, y_split)]),
        Polygon([(x_split, y_split), (maxx, y_split), (maxx, maxy), (x_split, maxy)]),
        Polygon([(minx, y_split), (x_split, y_split), (x_split, maxy), (minx, maxy)])
    ]

    subregion_errors = [[] for _ in range(4)]

    for traj_idx, traj in enumerate(x_y_trajectories):
        line = LineString(traj)
        error = all_perception_errors[traj_idx]

        for i, region in enumerate(subregions):
            intersection = line.intersection(region)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    coords = np.array(intersection.coords)
                    for pt in coords:
                        match = np.where((traj == pt).all(axis=1))[0]
                        if len(match) > 0:
                            start_idx = match[0]
                            break
                    else:
                        continue
                    end_idx = start_idx + len(coords)
                    subregion_errors[i].append(error[start_idx:end_idx])
                elif intersection.geom_type == 'MultiLineString':
                    for part in intersection.geoms:
                        coords = np.array(part.coords)
                        for pt in coords:
                            match = np.where((traj == pt).all(axis=1))[0]
                            if len(match) > 0:
                                start_idx = match[0]
                                break
                        else:
                            continue
                        end_idx = start_idx + len(coords)
                        subregion_errors[i].append(error[start_idx:end_idx])

    # Compute max errors and 95% bounds
    subregion_max_errors = [[] for _ in range(4)]
    for i in range(4):
        for err_segment in subregion_errors[i]:
            subregion_max_errors[i].append(np.max(err_segment))

    confidence_bounds = []
    for i in range(4):
        errors = subregion_max_errors[i].copy()
        errors.append(10000)
        errors_sorted = sorted(errors)
        index_95 = int(0.95 * len(errors_sorted))
        bound_95 = errors_sorted[index_95]
        confidence_bounds.append(bound_95)

    return sum(confidence_bounds)


# Genetic Algorithm
def genetic_algorithm(x_y_trajectories, all_perception_errors, bounds, pop_size=20, generations=30, mutation_rate=0.2):
    minx, miny, maxx, maxy = bounds

    # Initialize population: list of (x_split, y_split)
    population = [(random.uniform(minx, maxx), random.uniform(miny, maxy)) for _ in range(pop_size)]

    for gen in range(generations):
        fitnesses = [fitness_function(x, y, bounds, x_y_trajectories, all_perception_errors) for x, y in population]
        sorted_pop = [p for _, p in sorted(zip(fitnesses, population))]
        population = sorted_pop[:pop_size // 2]  # Select top half

        # Crossover: generate new individuals
        offspring = []
        while len(offspring) < pop_size - len(population):
            parent1, parent2 = random.sample(population, 2)
            child_x = (parent1[0] + parent2[0]) / 2
            child_y = (parent1[1] + parent2[1]) / 2
            offspring.append((child_x, child_y))

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                dx = random.uniform(-0.05, 0.05) * (maxx - minx)
                dy = random.uniform(-0.05, 0.05) * (maxy - miny)
                mutated = (np.clip(offspring[i][0] + dx, minx, maxx),
                           np.clip(offspring[i][1] + dy, miny, maxy))
                offspring[i] = mutated

        population.extend(offspring)

    # Final best
    best_fitness = float('inf')
    best_split = None
    for x, y in population:
        f = fitness_function(x, y, bounds, x_y_trajectories, all_perception_errors)
        if f < best_fitness:
            best_fitness = f
            best_split = (x, y)

    return best_split, best_fitness



data_filename = "./f110_dataset_06_26_2025.pkl"

with open(data_filename, 'rb') as f:
    all_trajectory_dict = pickle.load(f)
for key in all_trajectory_dict.keys():
    print(key)

all_state_trajectories = all_trajectory_dict['RealStates']
all_perception_errors = all_trajectory_dict['1DErrors']

x_y_trajectories = [traj[:, :2] for traj in all_state_trajectories]


all_points = np.vstack(x_y_trajectories)
multi_point = MultiPoint(all_points)
minx, miny, maxx, maxy = multi_point.envelope.bounds
bounds = (minx, miny, maxx, maxy)


best_split, best_loss = genetic_algorithm(x_y_trajectories, all_perception_errors, bounds)
print("Best x_split, y_split:", best_split)
print("Best total 95% bound loss:", best_loss)

####  Draw the figure for visulization  ##########
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint

all_points = np.vstack(x_y_trajectories)
multi_point = MultiPoint(all_points)
bounding_box = multi_point.envelope  # returns the smallest rectangle (aligned with axes)

minx, miny, maxx, maxy = bounding_box.bounds
bounding_box_coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]

# Unpack best split
x_split, y_split = best_split

# Define subregion rectangles
subregion_rects = [
    [(minx, miny), (x_split, miny), (x_split, y_split), (minx, y_split), (minx, miny)],  # bottom-left
    [(x_split, miny), (maxx, miny), (maxx, y_split), (x_split, y_split), (x_split, miny)],  # bottom-right
    [(x_split, y_split), (maxx, y_split), (maxx, maxy), (x_split, maxy), (x_split, y_split)],  # top-right
    [(minx, y_split), (x_split, y_split), (x_split, maxy), (minx, maxy), (minx, y_split)]   # top-left
]

# Start plotting
plt.figure(figsize=(10, 10))
for traj in x_y_trajectories:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.1)

# Draw bounding box
box_x, box_y = zip(*bounding_box_coords)
plt.plot(box_x, box_y, color='red', linewidth=2, label='Bounding Box')

# Draw subregion rectangles
colors = ['blue', 'green', 'orange', 'purple']
for i, rect in enumerate(subregion_rects):
    xs, ys = zip(*rect)
    plt.plot(xs, ys, linestyle='--', linewidth=2, label=f'Subregion {i+1}', color=colors[i])

# Annotate x_split and y_split lines
plt.axvline(x=x_split, color='black', linestyle=':', linewidth=1)
plt.axhline(y=y_split, color='black', linestyle=':', linewidth=1)

plt.title("Optimized Subregions within Bounding Rectangle")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()




#### Quick test ######################################

import matplotlib.pyplot as plt
import gzip
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import split as split_geom
import numpy as np

def main():
    data_filename = "./f110_dataset_06_26_2025.pkl"
    
    with open(data_filename, 'rb') as f:
        all_trajectory_dict = pickle.load(f)
    for key in all_trajectory_dict.keys():
        print(key)
    
    all_state_trajectories = all_trajectory_dict['RealStates']
    all_perception_errors = all_trajectory_dict['1DErrors']

    x_y_trajectories = [traj[:, :2] for traj in all_state_trajectories]

    ### Find the smallest rectangle ###
    from shapely.geometry import MultiPoint
    all_points = np.vstack(x_y_trajectories)
    multi_point = MultiPoint(all_points)
    bounding_box = multi_point.envelope  # returns the smallest rectangle (aligned with axes)

    # Extract bounds
    minx, miny, maxx, maxy = bounding_box.bounds
    bounding_box_coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]

    # plt.figure(figsize=(10, 10))
    # for traj in x_y_trajectories:
    #     plt.plot(traj[:, 0], traj[:, 1], alpha=0.1)
    #
    # # Plot bounding box
    # box_x, box_y = zip(*bounding_box_coords)
    # plt.plot(box_x, box_y, color='red', linewidth=2, label='Bounding Box')
    #
    # plt.title("All Trajectories with Bounding Rectangle")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.legend()
    # plt.show()


##### split the rectangle ######
    # Step 1: Compute bounding box
    all_points = np.vstack(x_y_trajectories)
    multi_point = MultiPoint(all_points)
    minx, miny, maxx, maxy = multi_point.envelope.bounds

    # Define one vertical and one horizontal line to split into 4 subregions
    x_split = minx + (maxx - minx) / 2
    y_split = miny + (maxy - miny) / 2

    # Define 4 rectangular subregions
    subregions = [
        Polygon([(minx, miny), (x_split, miny), (x_split, y_split), (minx, y_split)]),  # bottom-left
        Polygon([(x_split, miny), (maxx, miny), (maxx, y_split), (x_split, y_split)]),  # bottom-right
        Polygon([(x_split, y_split), (maxx, y_split), (maxx, maxy), (x_split, maxy)]),  # top-right
        Polygon([(minx, y_split), (x_split, y_split), (x_split, maxy), (minx, maxy)])  # top-left
    ]

    # Step 2: For each subregion, extract the parts of the trajectories inside it
    # Initialize separate lists for trajectories and errors for each subregion
    subregion_trajectories = [[] for _ in range(4)]
    subregion_errors = [[] for _ in range(4)]

    for traj_idx, traj in enumerate(x_y_trajectories):
        line = LineString(traj)
        error = all_perception_errors[traj_idx]  # shape: (T,) or (T,1)

        for i, region in enumerate(subregions):
            intersection = line.intersection(region)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    coords = np.array(intersection.coords)
                    # Find corresponding indices in original trajectory
                    for pt in coords:
                        match = np.where((traj == pt).all(axis=1))[0]
                        if len(match) > 0:
                            start_idx = match[0]
                            break
                    else:
                        continue  # No match found, skip

                    end_idx = start_idx + len(coords)
                    subregion_trajectories[i].append(coords)
                    subregion_errors[i].append(error[start_idx:end_idx])

                elif intersection.geom_type == 'MultiLineString':
                    for part in intersection.geoms:
                        coords = np.array(part.coords)
                        for pt in coords:
                            match = np.where((traj == pt).all(axis=1))[0]
                            if len(match) > 0:
                                start_idx = match[0]
                                break
                        else:
                            continue

                        end_idx = start_idx + len(coords)
                        subregion_trajectories[i].append(coords)
                        subregion_errors[i].append(error[start_idx:end_idx])

    # Store the max error per trajectory segment for each subregion
    subregion_max_errors = [[] for _ in range(4)]

    for i in range(4):  # iterate over subregions
        for err_segment in subregion_errors[i]:
            max_error = np.max(err_segment)
            subregion_max_errors[i].append(max_error)



### calculate the confidence bounds.###

    confidence_bounds = []

    for i in range(4):
        errors = subregion_max_errors[i].copy()
        errors.append(10000)  # add a large outlier
        errors_sorted = sorted(errors)

        index_95 = int(0.95 * len(errors_sorted))
        bound_95 = errors_sorted[index_95]

        confidence_bounds.append(bound_95)

    # Print or use the 95% confidence bounds
    for i, bound in enumerate(confidence_bounds):
        print(f"Subregion {i}: 95% confidence bound = {bound}")


    # take half for conformal prediction, half for testing (repeated from last time)
    # random.seed(0)
    # conf_fraction = 0.5
    # conf_traj_inds = random.sample(range(len(all_state_trajectories)), int(conf_fraction*len(all_state_trajectories)))
    # test_traj_inds = list(set(range(len(all_state_trajectories)))-set(conf_traj_inds))
    #
    # conf_state_trajectories = [all_state_trajectories[i] for i in conf_traj_inds]
    # conf_perception_errors = [all_perception_errors[i] for i in conf_traj_inds]
    #
    # test_state_trajectories = [all_state_trajectories[i] for i in test_traj_inds]
    # test_perception_errors = [all_perception_errors[i] for i in test_traj_inds]
    #
    # ## data extracted, now get regional conformal bounds
    # print("Total Trajectories in dataset: {}".format(len(all_state_trajectories)))
    # print("Number of Trajectories for deriving conformal bounds: {}".format(len(conf_state_trajectories)))
    # print("Number of Trajectories for testing bounds: {}".format(len(test_state_trajectories)))

if __name__ == '__main__':
    main()