import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP import get_MDP
from MDP import dynamic_programming_finite_horizon
from MDP import get_StateBounds
from MDP import state_from_obs
import sys
import pickle



if __name__ == '__main__':

    np.random.seed(1)

    num_data_per_state = 50
    env = gym.make('MountainCarContinuous-v0').env.unwrapped

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    num_acts = 4
    MDP_state_bounds = [[pos_min, pos_max],[vel_min, vel_max]]
    action_space_env = [act_min+(act_max-act_min)*i/num_acts for i in range(num_acts+1)]
    # discretizing action space? ----------------------------------------------------------------------------------------------------
    # becomes [-1, -0.5, 0, 0.5, 1] for num_acts = 4

    # if you don't want to rebuild your MDP every time, you
    # can save/load it as a pickle, for example
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'rb') as f:
            MC_MDP = pickle.load(f)

    else:

        # this is just a placeholder
        MC_MDP = get_MDP(env, MDP_state_bounds, action_space_env, num_data_per_state)

        
        filename = 'mc_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(MC_MDP, f, pickle.HIGHEST_PROTOCOL)

    
    gamma = 1
    T = 150
    num_episodes = 1000
    num_bins = int(np.sqrt(len(MC_MDP.keys())))

    state_bounds_dict = get_StateBounds(MDP_state_bounds, num_bins)
    pi = dynamic_programming_finite_horizon(MC_MDP, action_space_env, gamma, T)

    perception_error_bound = (-0.02961, 0.02180)

    # YOUR CODE GOES HERE
    print("Evaluating Policy...")
    # evaluate policy
    all_total_rewards = []
    
    success = 0
    fail = 0

    for __ in range(num_episodes):

        env.reset()
        total_reward = 0
        all_rewards = []
        observation = env.state

        positions = []
        
        for t in range(T):
            observation[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
            state = state_from_obs(observation, state_bounds_dict)
            print(f'{state},{t}')
            """
            if state < 5:
                plt.plot(positions)
                break
            """
            positions.append(observation[0])
            #positions.append(state_bounds_dict[state][0][0])
            #print(state_bounds_dict)
            #state[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
            #print(state)
            action = action_space_env[pi[(state, t)]]
            #print(action)
            observation, reward, done, __, ___ = env.step([action])
            total_reward += reward
            all_rewards.append(reward)
            #print(reward)
            if done or t == T-1 or state < 10:
                #print(total_reward)

                if t == T-1 or state < 10:
                    fail += 1
                else:
                    success += 1

                all_total_rewards.append(total_reward)

                plt.plot(positions)
                #print("Success!")
                break
                #all_rewards.append(reward)
                #total_reward += sum(all_rewards)
                #all_total_rewards.append(total_reward/T)
        
    
    print(success / (success + fail))
    plt.show()
    print("Average Reward over " + str(num_episodes) +" episodes:  " + str(sum(all_total_rewards)/num_episodes))
    


