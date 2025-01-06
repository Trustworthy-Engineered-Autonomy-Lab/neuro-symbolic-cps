import matplotlib.pyplot as plt
import gzip
import math
import random
import pickle

import numpy as np


def get_traj_subset_bounds(est_trajs, real_trajs, conf=0.99, bounds=None, num_bins=10):
    if bounds is None:
        max_val = .6
        min_val = -1.2
        interval_bounds = np.linspace(min_val, max_val, num_bins+1)
        bounds = np.concatenate([interval_bounds[:-1].reshape(num_bins,1), interval_bounds[1:].reshape(num_bins, 1)], axis=1)
    
    all_regional_traj_bounds = []
    for bound in bounds:
        all_traj_bounds = []

        for est_traj, real_traj in zip(est_trajs, real_trajs):
            in_set = (real_traj >= bound[0]) * (real_traj  < bound[1])

            region_indices = np.where(in_set)[0]
            if len(region_indices) == 0:
                traj_bounds = np.array([np.nan, np.nan])
            else:
                traj_errors = est_traj[region_indices] - real_traj[region_indices] # errors for each timestep in the region
                abs_traj_errors = np.abs(traj_errors)
                traj_bounds = np.array([-np.max(abs_traj_errors), np.max(abs_traj_errors)])
                if traj_bounds[0] > 0:
                    traj_bounds[0] = 0
                if traj_bounds[1] < 0:
                    traj_bounds[1] = 0
            all_traj_bounds.append(traj_bounds)
        all_regional_traj_bounds.append(all_traj_bounds)
    
    # now we have a list of max diffs over each trajectory in each region.
    # we want a bound for each region, such that conf level of trajectories are covered from each region

    
    regional_conf_bounds = []
    for bound, regional_traj_bounds in zip(bounds, all_regional_traj_bounds):
        num_traj_needed = math.ceil(conf*len(regional_traj_bounds))

        # sort the bounds by size, then take the num_needed'th of them
        bounds_sizes = np.array([b[1]-b[0] for b in regional_traj_bounds])

        if np.any(bounds_sizes < 0):
            print("negative bound size, failure")
            break

        sorted_bound_sizes = np.sort(bounds_sizes)
        sorted_inds = np.argsort(bounds_sizes)
        sorted_regional_traj_bounds = [regional_traj_bounds[i] for i in sorted_inds]
        # take the smallest bounds that cover conf percent of trajectories
        conf_bounds = sorted_regional_traj_bounds[num_traj_needed-1]
        regional_conf_bounds.append((bound, conf_bounds))

        print("For x region [{:.4f}, {:.4f}], {:.4f}% Trajectory Conformal bounds are [{:.5f}, {:.5f}]".format(bound[0], bound[1], 100*conf, conf_bounds[0], conf_bounds[1]))   
    
    return regional_conf_bounds

def get_traj_noise_subset_bounds(est_trajs, real_trajs, conds, conf=0.99, bounds=None, num_bins=10, noise_ind=0):
    if bounds is None:
        if noise_ind == 0:
            max_val = 2
            min_val = .1
        elif noise_ind == 1:

            max_val = 0.075
            min_val = 0

        interval_bounds = np.linspace(min_val, max_val, num_bins+1)
        bounds = np.concatenate([interval_bounds[:-1].reshape(num_bins,1), interval_bounds[1:].reshape(num_bins, 1)], axis=1)
    
    all_regional_traj_bounds = []
    last_indices = set()
    for bound in bounds:
        all_traj_bounds = []
        in_set = (conds[:,noise_ind] >= bound[0]) * (conds[:,noise_ind]  <= bound[1])
        region_indices = set(np.where(in_set)[0])
        region_indices = region_indices - last_indices
        last_indices= region_indices
        region_indices = list(region_indices)
        regional_ests = [est_trajs[i] for i in region_indices]
        regional_reals = [real_trajs[i] for i in region_indices]
        for est_traj, real_traj in zip(regional_ests, regional_reals): # this is just trajectories with desired noise params

            #print(real_traj[region_indices])
            if len(region_indices) == 0:
                traj_bounds = np.array([np.nan, np.nan])
            else:
                traj_errors = est_traj- real_traj # errors for each timestep in the region
                #traj_bounds = np.array([np.min(traj_errors), np.max(traj_errors)])
                abs_traj_errors = np.abs(traj_errors)
                traj_bounds = np.array([-np.max(abs_traj_errors), np.max(abs_traj_errors)])
                if traj_bounds[0] > 0:
                    traj_bounds[0] = 0
                if traj_bounds[1] < 0:
                    traj_bounds[1] = 0
            all_traj_bounds.append(traj_bounds)
        all_regional_traj_bounds.append(all_traj_bounds)
    
    # now we have a list of max diffs over each trajectory in each region.
    # we want a bound for each region, such that conf level of trajectories are covered from each region

    
    regional_conf_bounds = []
    for bound, regional_traj_bounds in zip(bounds, all_regional_traj_bounds):
        num_traj_needed = math.ceil(conf*len(regional_traj_bounds))

        # sort the bounds by size, then take the num_needed'th of them
        bounds_sizes = np.array([b[1]-b[0] for b in regional_traj_bounds])

        if np.any(bounds_sizes < 0):
            print("negative bound size, failure")
            break

        sorted_bound_sizes = np.sort(bounds_sizes)
        sorted_inds = np.argsort(bounds_sizes)
        sorted_regional_traj_bounds = [regional_traj_bounds[i] for i in sorted_inds]
        # take the smallest bounds that cover conf percent of trajectories
        if(len(sorted_regional_traj_bounds) == 0):
            conf_bounds = np.array([np.nan, np.nan])
        else:
            conf_bounds = sorted_regional_traj_bounds[num_traj_needed-1]
        regional_conf_bounds.append((bound, conf_bounds))

        if noise_ind == 1:
            print("For Blur values [{:.4f}, {:.4f}], {:.1f}% Trajectory Conformal bounds are [{:.5f}, {:.5f}]".format(bound[0], bound[1], 100*conf, conf_bounds[0], conf_bounds[1]))
        elif noise_ind == 0:
            print("For Contrast values [{:.4f}, {:.4f}], {:.1f}% Trajectory Conformal bounds are [{:.5f}, {:.5f}]".format(bound[0], bound[1], 100*conf, conf_bounds[0], conf_bounds[1]))
    return regional_conf_bounds

def check_traj_noise_subset_bounds(est_trajs, real_trajs, conds, conf_bounds, conf, bounds, noise_ind=0):

    all_regional_accs = []
    last_indices = set()
    for bound, cb in zip(bounds, conf_bounds):
        num_correct = 0
        num_traj = 0
        num_missing = 0
        in_set = (conds[:,noise_ind] >= bound[0]) * (conds[:,noise_ind]  <= bound[1])
        region_indices = set(np.where(in_set)[0])
        region_indices = region_indices - last_indices
        last_indices= region_indices
        region_indices = list(region_indices)
        regional_ests = [est_trajs[i] for i in region_indices]
        regional_reals = [real_trajs[i] for i in region_indices]
        for est_traj, real_traj in zip(regional_ests, regional_reals): # there are 2500/2=1250 trajectories per region

            # find the errors over the region as determined by the real positions
            if len(region_indices) == 0:
                num_traj += 1
                num_missing +=1
            else:
                traj_errors = est_traj - real_traj # errors for each timestep in the region
                #print(np.max(traj_errors), cb[1])
                #print(np.min(traj_errors), cb[0])
                num_correct += int((np.max(traj_errors) <= cb[1]) and (np.min(traj_errors)>=cb[0])) # checks over whole trajectory
                num_traj+=1
        if num_missing == num_traj:
            test_acc = np.nan
        else:
            test_acc = num_correct/num_traj
        all_regional_accs.append(test_acc)


        
        if noise_ind == 1:
            print("For Blur values [{:.4f}, {:.4f}], {:.1f}% Trajectory Conformal bounds hold for {:.4f}% of test trajectories".format(bound[0], bound[1], 100*conf, 100*test_acc))
        elif noise_ind == 0:
            print("For Contrast values [{:.4f}, {:.4f}], {:.1f}% Trajectory Conformal bounds hold for {:.4f}% of test trajectories".format(bound[0], bound[1], 100*conf, 100*test_acc))
    
    return all_regional_accs

def check_traj_subset_bounds(est_trajs, real_trajs, conf_bounds, conf, bounds):

    all_regional_accs = []
    for bound, cb in zip(bounds, conf_bounds):
        num_correct = 0
        num_traj = 0
        num_missing = 0
        for est_traj, real_traj in zip(est_trajs, real_trajs): # there are 2500/2=1250 trajectories per region

            # find the errors over the region as determined by the real positions
            # generate dataset and loader for conformal over desired region

            in_set = (real_traj >= bound[0]) * (real_traj  < bound[1])
            
            region_indices = np.where(in_set)[0]
            if len(region_indices) == 0:
                num_missing +=1
                num_traj += 1
            else:
                traj_errors = est_traj[region_indices] - real_traj[region_indices] # errors for each timestep in the region
                #print(np.max(traj_errors), cb[1])
                #print(np.min(traj_errors), cb[0])
                num_correct += int((np.max(traj_errors) <= cb[1]) and (np.min(traj_errors)>=cb[0])) # checks over whole trajectory
                num_traj+=1
        if num_missing == num_traj:
            test_acc = np.nan
        else:
            test_acc = num_correct/num_traj
        all_regional_accs.append(test_acc)


        
        print("For x region [{:.4f}, {:.4f}], {:.4f}% Trajectory Conformal bounds hold for {:.4f}% of test trajectories".format(bound[0], bound[1], 100*conf, 100*test_acc))

    
    return all_regional_accs

def main():
    #data_filename = "TrajData_NonRobust_NumCond5_NumEp100_AlphaMin0.5_AlphaMax1.5_DeltaMin0.0_DeltaMax0.05.pkl"
    data_filename = "TrajData_robust_nn_NumCond5_NumEp100_AlphaMin0.5_AlphaMax1.5_DeltaMin0.0_DeltaMax0.02.pkl"
    
    with open(data_filename, 'rb') as f:
        all_trajectory_dicts = pickle.load(f)
    all_est_trajectories = []
    all_real_trajectories = []
    all_conds = []
    for td in all_trajectory_dicts:
        if (len(td["EstTrajectories"]) != len(td["RealTrajectories"])):
            print("Different number of trajectories between real and estimated")
            break
        cond = np.array(td["Conditions"])
        for i in range(len(td["EstTrajectories"])):
            traj_est = np.array(td["EstTrajectories"][i])
            traj_real = np.array(td["RealTrajectories"][i])
            all_est_trajectories.append(traj_est)
            all_real_trajectories.append(traj_real)
            all_conds.append(cond)

    # take half for conformal prediction, half for testing
    random.seed(0)
    conf_fraction = 0.5
    conf_traj_inds = random.sample(range(len(all_est_trajectories)), int(conf_fraction*len(all_est_trajectories)))
    test_traj_inds = list(set(range(len(all_est_trajectories)))-set(conf_traj_inds))

    conf_est_trajectories = [all_est_trajectories[i] for i in conf_traj_inds]
    conf_real_trajectories = [all_real_trajectories[i] for i in conf_traj_inds]
    conf_conds = np.array([all_conds[i] for i in conf_traj_inds])

    test_est_trajectories = [all_est_trajectories[i] for i in test_traj_inds]
    test_real_trajectories = [all_real_trajectories[i] for i in test_traj_inds]
    test_conds = np.array([all_conds[i] for i in test_traj_inds])

    ## data extracted, now get regional conformal bounds
    print("Total Trajectories in dataset: {}".format(len(all_est_trajectories)))
    print("Number of Trajectories for deriving conformal bounds: {}".format(len(conf_est_trajectories)))
    print("Number of Trajectories for testing bounds: {}".format(len(test_est_trajectories)))



    # show how it works for different bins and conf levels    
    delta = .2
    mids1 = np.linspace(-1.2+delta, .6-delta, num=9)
    mids2 = np.linspace(-1.2+delta+delta, .6-delta-delta, num=9)
    mids1 = [-1, -.6, -.65, -.5, -.4, -.3, -.2, -.1, 0, .2]
    mids2 = [-.55,-.5, -.45, -.4, -.35, -.3, -.25, -.2]
    A = 0.03
    n = 3
    #mid = -1.0
    conf_values = [1-A]
    num_bins = [1] 
    for conf_value in conf_values:
        for nb in num_bins:

            print("Finding {:.1f}% Trajectory Conformal Bounds for position in {} fixed-width bins...".format(100*conf_value, nb))
            
            
            regional_out = get_traj_subset_bounds(conf_est_trajectories, conf_real_trajectories, conf=conf_value, bounds=None, num_bins=nb)
            bounds = [out[0] for out in regional_out]
            conf_bounds = [out[1] for out in regional_out]
            print()
            print("Checking Conformal Bounds on test set...")
            test_accs = check_traj_subset_bounds(test_est_trajectories, test_real_trajectories, conf_bounds, conf_value, bounds)
            print("Minimum accuracy over all regions: {:.2f}".format(100*min(test_accs)))
            print()
        print("\n")
    for mid1 in mids1:
        for mid2 in mids2:
            if mid1 >= mid2:
                continue

            

            bounds = [[-1.2, mid1], [mid1, mid2], [mid2, .6]]
            #bounds = [[-1.2, mid1], [mid1, .6]]
            bounds = [[-1.2000,  -0.57500],
                        [-0.57500,  -.1],
                        [-.1,  0.6000]]
            n = len(bounds)
            conf_values = [1-A/n]
            num_bins = [n] 
            for conf_value in conf_values:
                for nb in num_bins:

                    print("Finding {:.1f}% Trajectory Conformal Bounds for position in {} fixed-width bins...".format(100*conf_value, nb))
                    
                    
                    regional_out = get_traj_subset_bounds(conf_est_trajectories, conf_real_trajectories, conf=conf_value, bounds=bounds, num_bins=nb)
                    bounds = [out[0] for out in regional_out]
                    conf_bounds = [out[1] for out in regional_out]
                    print()
                    print("Checking Conformal Bounds on test set...")
                    test_accs = check_traj_subset_bounds(test_est_trajectories, test_real_trajectories, conf_bounds, conf_value, bounds)
                    print("Minimum accuracy over all regions: {:.2f}".format(100*min(test_accs)))
                    print()
                print("\n")
    ### This can work to show trajectory conformal bounds over noise ranges instead of state regions. It's commented out because 
    ### we are focused on state regions at the moment
    '''
    # noise_ind = 0 -> contrast level, 
    # noise_ind = 1 -> blur level, 
    noise_inds = [0,1]
    conf_values = [0.99]
    num_bins = [3, 10]
    for noise_ind in noise_inds:
        for conf_value in conf_values:
            for nb in num_bins:
                if noise_ind == 0:
                    print("Finding {:.1f}% Trajectory Conformal Bounds for Contrast in {} fixed-width bins...".format(100*conf_value, nb))
                elif noise_ind == 1:
                    print("Finding {:.1f}% Trajectory Conformal Bounds for Blur in {} fixed-width bins...".format(100*conf_value, nb))
                
                regional_out = get_traj_noise_subset_bounds(conf_est_trajectories, conf_real_trajectories, conf_conds, noise_ind=noise_ind, conf=conf_value, bounds=None, num_bins=nb)
                bounds = [out[0] for out in regional_out]
                conf_bounds = [out[1] for out in regional_out]
                print()
                print("Checking Conformal Bounds on test set...")
                test_accs = check_traj_noise_subset_bounds(test_est_trajectories, test_real_trajectories, test_conds, conf_bounds, conf_value, bounds, noise_ind=noise_ind)
                print("Minimum accuracy over all regions: {:.2f}".format(100*min(test_accs)))
                print()
            print("\n")
    '''


if __name__ == "__main__":
    main()