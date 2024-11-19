import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from MDP_Error import get_MDP
from MDP_Error import dynamic_programming_finite_horizon, policy_iteration, value_iteration
from MDP_Error import get_StateBounds
from MDP_Error import state_from_obs
import sys
import pickle

if __name__ == '__main__':
    
    load_MDP = input("Enter 1 for new MDP: ")
    load_PI = input("Enter 1 for new policy: ")
    num_episodes = int(input("Enter # of episodes: "))
    error_multiplier = int(input("Enter error bound multiplier: "))
    graph = int(input("Enter 1 for state graph, 2 for position vs time: "))

    if graph == 1:
      ax = plt.figure().add_subplot(1, 1, 1)

      # Major ticks every 20, minor ticks every 5
      x_major_ticks = np.arange(-1.2, 0.6, 0.18)
      x_minor_ticks = np.arange(-1.2, 0.6, 0.045)
      y_major_ticks = np.arange(-0.07, 0.07, 0.0105)
      y_minor_ticks = np.arange(-0.07, 0.07, 0.0035)

      ax.set_xticks(x_major_ticks)
      ax.set_yticks(y_major_ticks)
      ax.set_xticks(x_minor_ticks, minor=True)
      ax.set_yticks(y_minor_ticks, minor=True)

      # And a corresponding grid
      ax.grid(which='both')

      # Or if you want different settings for the grids:
      ax.grid(which='minor', alpha=0.2)
      ax.grid(which='major', alpha=0.5)

      plt.xlim(-1.2, 0.6)
      plt.ylim(-0.07, 0.07)

      plt.xlabel("Position")
      plt.ylabel("Velocity")
      plt.title("State Grid")
    elif graph == 2:
       plt.xlabel("Time")
       plt.ylabel("Position")
       plt.title("Position vs Time")

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
    #get_MDP(env, MDP_state_bounds, action_space_env, num_data_per_state)
    if load_MDP != '1':
      try:
            filename = './MDP/MDP_Errors/mc_mdp.pkl'
            with open(filename, 'rb') as f:
                MC_MDP = pickle.load(f)
      except:
            MC_MDP = get_MDP(env, MDP_state_bounds, action_space_env, num_data_per_state)
            filename = './MDP/MDP_Errors/mc_mdp.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(MC_MDP, f, pickle.HIGHEST_PROTOCOL)
    else:
        MC_MDP = get_MDP(env, MDP_state_bounds, action_space_env, num_data_per_state)
        filename = './MDP/MDP_Errors/mc_mdp.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(MC_MDP, f, pickle.HIGHEST_PROTOCOL)

    gamma = 1
    T = 110
    num_bins = int(np.sqrt(len(MC_MDP.keys())))

    state_bounds_dict = get_StateBounds(MDP_state_bounds, num_bins)

    if load_PI != '1':
      try:
          filename = './MDP/MDP_Errors/mc_policy.pkl'
          with open(filename, 'rb') as f:
              pi = pickle.load(f)
      except:
          pi = dynamic_programming_finite_horizon(MC_MDP, action_space_env, gamma, T)
    else:
       pi = dynamic_programming_finite_horizon(MC_MDP, action_space_env, gamma, T)

    perception_error_bound = (error_multiplier * -0.02961, error_multiplier * 0.02180)

    # YOUR CODE GOES HERE
    print("Evaluating Policy...")
    # evaluate policy
    all_total_rewards = []
    
    success = 0
    fail = 0

    bucket_diff = []


    for __ in range(num_episodes):

        env.reset()
        total_reward = 0
        all_rewards = []
        observation = env.state

        positions = []

        prev_position = 0
        prev_velocity = 0
        velocity = 0
        perception_differences = 0
        
        for t in range(T):
            
            temp_state = state_from_obs(observation, state_bounds_dict)

            true_observation = observation[0]
            true_velocity = observation[1]

            observation[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
            #observation[0] += np.random.normal(-0.005, 0.004)

            if t == 0:
                velocity = 0
            else:
                #velocity = observation[0] - prev_position
                prev_velocity = velocity
                velocity = prev_velocity + action * 0.0015 - 0.0025 * np.cos(3 * observation[0])
            
          
            prev_position = observation[0]

            if (observation[0] < -1.2):
                observation[0] = -1.2
            elif (observation[0] > 0.6):
                observation[0] = 0.6
            
            if (velocity < vel_min):
                velocity = vel_min
            elif (velocity > vel_max):
                velocity = vel_max

            observation[1] = velocity

            if graph == 1:
              plt.plot(observation[0], observation[1], 'o', color="orange")
              plt.plot(true_observation, true_velocity, 'o', color="blue")
              plt.plot([observation[0], true_observation], [observation[1], true_velocity], color='black')

            state = state_from_obs(observation, state_bounds_dict)

            if temp_state != state:
                perception_differences += 1

            #print(f'{state},{t}')

            action = action_space_env[pi[(state, t)]]
            
            positions.append(observation[0])
            observation[0] = true_observation
            

            observation, reward, done, __, ___ = env.step([action])
            total_reward += reward
            all_rewards.append(reward)
            #print(reward)
            if done or t == T-1: #or state < 10:
                #print(total_reward)

                if t == T-1: #or state < 10:
                    fail += 1
                    if graph == 2:
                      plt.plot(positions)
                else:
                    success += 1

                all_total_rewards.append(total_reward)

                if graph == 2:  
                  plt.plot(positions)
                bucket_diff.append(perception_differences)
                #print("Success!")
                break
                #all_rewards.append(reward)
                #total_reward += sum(all_rewards)
                #all_total_rewards.append(total_reward/T)
            
        
    
    success_rate = success / (success + fail)
    print(f'Success %: {success_rate}')

    total = 0
    for i in range(len(bucket_diff)):
        total += bucket_diff[i]
    average = total / len(bucket_diff)
    print(f'Average # of state differences: {average}')
    print(f'Error bound:{perception_error_bound}')

    if success_rate == 1.0 and load_PI == '1':
        filename = './MDP/MDP_Errors/mc_policy.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
    
    
    plt.show()
    print("Average Reward over " + str(num_episodes) +" episodes:  " + str(sum(all_total_rewards)/num_episodes))
    


