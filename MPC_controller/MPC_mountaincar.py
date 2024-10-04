import gym
import numpy as np
from scipy.optimize import minimize
print("NumPy version:", np.__version__)
print("Gym version:", gym.__version__)

import matplotlib.pyplot as plt
import matplotlib
print("matplotlib version:", matplotlib.__version__)

import time

start_time = time.time()
# Define the MPC parameters
N = 110  # Prediction horizon
lambda_control = 0.0005  # Control effort penalty

# Define the MountainCarContinuous environment
env = gym.make('MountainCarContinuous-v0')
env.reset()


# Define the dynamics of the system (taken from the MountainCarContinuous env)
def dynamics(state, action):
    position, velocity = state
    force = action[0]
    velocity = np.clip(velocity + force * 0.0015 - np.cos(3 * position) * 0.0025, -0.07, 0.07)
    position = np.clip(position + velocity, -1.2, 0.6)
    if position == -1.2 and velocity < 0:
        velocity = 0
    return np.array([position, velocity])


# Define the objective function for MPC
def objective(actions, initial_state):
    state = np.copy(initial_state)
    cost = 0.0
    for i in range(N-30, N):
        print(i)
        action = actions[i:i + 1]
        state = dynamics(state, action)
        position, _ = state
        cost += (position - 0.45) ** 2 + lambda_control * action ** 2  # Distance to goal and control penalty
    return cost


# Define the MPC loop
def mpc_controller(state):
    # Initialize control inputs (forces) for the horizon
    initial_guess = np.zeros(N)

    # Optimize the sequence of actions
    result = minimize(objective, initial_guess, args=(state,), bounds=[(-1, 1)] * N, method='SLSQP')

    # Return the first action
    return result.x[0]


# Run the MPC in the environment
done = False
state_list = []
state, others = env.reset()
for _ in range(200):
    env.render()
    action_list = mpc_controller(state)
    action = np.array([action_list])
    state, reward, done, truncation, info = env.step(action)
    state_list.append(state)

    if done:
        print("Succeed!")
        break

env.close()
end_time = time.time()
print("the total running time is ", end_time - start_time)


position = [item[0] for item in state_list]
plt.plot(position)
plt.title("State List Plot")
plt.xlabel("Index")
plt.ylabel("Position Value")
plt.show()
