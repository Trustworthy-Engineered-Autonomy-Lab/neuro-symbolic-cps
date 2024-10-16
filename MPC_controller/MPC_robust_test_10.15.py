import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import random



def Robust_MPC_two_error(x0, error_bound, penalty_start, N):
    # Create non-convex optimization problem
    pos1 = x0[0, 0]; pos2 = x0[0, 1]; pos3 = x0[0, 2]
    vel1 = x0[1,0]; vel2 = x0[1,1]; vel3 = x0[1,2]
    pos_lb = error_bound[0, 0]; pos_ub = error_bound[0, 1]
    vel_lb = error_bound[1, 0]; vel_ub = error_bound[1, 1]

    opti = ca.Opti()

    # Variables
    x = opti.variable(2, N + 1)  # State trajectory (position and velocity)
    u = opti.variable(1, N)  # Control trajectory (force)

    # Reward variable (no need to define this as an optimization variable, we will add it directly to the cost)
    reward_value = -1000  # Large reward for reaching the goal (negative cost)

    # Initialize the cost to 0
    cost = 0

    # First three known postion and their error bound
    pos1_bound = [pos1 - pos_lb, pos1 + pos_ub]
    pos2_bound = [pos2 - pos_lb, pos2 - pos_ub]
    pos3_bound = [pos3 - pos_lb, pos3 - pos_ub]

    # Apply the constraints on the first three positions
    opti.subject_to(x[0, 0] >= pos1_bound[0])  # First state lower bound
    opti.subject_to(x[0, 0] <= pos1_bound[1])  # First state upper bound

    opti.subject_to(x[0, 1] >= pos2_bound[0])  # Second state lower bound
    opti.subject_to(x[0, 1] <= pos2_bound[1])  # Second state upper bound

    opti.subject_to(x[0, 2] >= pos3_bound[0])  # Third state lower bound
    opti.subject_to(x[0, 2] <= pos3_bound[1])  # Third state upper bound

    # First three known velocity and their error bounds
    vel1_bound = [vel1 - vel_lb, vel1 + vel_ub]
    vel2_bound = [vel2 - vel_lb, vel2 + vel_ub]
    vel3_bound = [vel3 - vel_lb, vel3 + vel_ub]

    opti.subject_to(x[1, 0] >= vel1_bound[0])  # First state lower bound
    opti.subject_to(x[1, 0] <= vel1_bound[1])  # First state upper bound

    opti.subject_to(x[1, 1] >= vel2_bound[0])  # Second state lower bound
    opti.subject_to(x[1, 1] <= vel2_bound[1])  # Second state upper bound

    opti.subject_to(x[1, 2] >= vel3_bound[0])  # Third state lower bound
    opti.subject_to(x[1, 2] <= vel3_bound[1])  # Third state upper bound

    # opti.subject_to(x[1, 0] == 0)
    # opti.subject_to(x[1, 1] == -0.00168)
    # opti.subject_to(x[1, 2] == -0.00334)

    # Define the dynamics and cost function
    for k in range(2, N):
        # Mountain Car dynamics
        position_k = x[0, k]
        velocity_k = x[1, k]
        force_k = u[0, k]

        # Apply the Mountain Car dynamics model
        velocity_next = velocity_k + (0.0015 * force_k - 0.0025 * ca.cos(3 * position_k))
        position_next = position_k + velocity_next

        # Apply dynamics constraints
        opti.subject_to(x[0, k + 1] == position_next)
        opti.subject_to(x[1, k + 1] == velocity_next)

        # Reward for reaching the goal
        cost += reward_value * ca.fmax(0, x[0, k] - 0.45)  # Add reward if position exceeds 0.45
        # Add penalty for control effort (minimize control action)
        cost += 0.01 * (force_k ** 2)  # Penalize large control actions

    # Add a cost for the last 30 states to be as close to the goal set (0.45) as possible
    for k in range(penalty_start, N + 1):
        cost += 10 * (x[0, k] - 0.45) ** 2  # Penalize deviation from the goal

    # Set the objective to minimize cost (which now includes both the reward and the last 30-state penalty)
    opti.minimize(cost)

    # State constraints
    for k in range(N):
        opti.subject_to(x[0, k] >= -1.2)  # Minimum position
        opti.subject_to(x[0, k] <= 0.6)  # Maximum position
        opti.subject_to(x[1, k] >= -0.07)  # Minimum velocity
        opti.subject_to(x[1, k] <= 0.07)  # Maximum velocity

        # Control input bounds
        opti.subject_to(u[0, k] >= -1.0)
        opti.subject_to(u[0, k] <= 1.0)

    # Initial state constraint
    #opti.subject_to(x[:, 0] == x0)

    # Set solver (IPOPT for nonlinear problems)
    opti.solver('ipopt')

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the solution
    x_opt = sol.value(x)
    u_opt = sol.value(u)

    return x_opt, u_opt

class MountainCarDynamics:
    def __init__(self):
        self.gravity = 0.0025
        self.force = 0.0015  # Updated force value
        self.max_speed = 0.07
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.45

    def step(self, state, action):
        """
        Parameters:
        - state: A tuple of (position, velocity)
        - action: A scalar value representing the action force to apply

        Returns:
        - next_state: Updated (position, velocity) after applying the action
        """
        position, velocity = state
        action = np.clip(action, -1, 1)  # Ensure action is within [-1, 1]

        # Update the velocity
        velocity += (action * self.force) - (self.gravity * np.cos(3 * position))
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)

        # Update the position
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        # Check if the car has reached the left boundary
        if position == self.min_position and velocity < 0:
            velocity = 0

        next_state = (position, velocity)
        return next_state


success_result = set(); fail_result = set()
for i in range(100):
    # 1. generate the exact states x1, v1, u1, u2
    initial_position_range = (-0.6, -0.4)
    x1 = np.random.uniform(initial_position_range[0], initial_position_range[1])
    init_velocity_range = (-0.01, 0.01)
    v1 = np.random.uniform(init_velocity_range[0], init_velocity_range[1])
    torque_range = (-1, 1)
    u1 = np.random.uniform(torque_range[0], torque_range[1]); u2 = np.random.uniform(torque_range[0], torque_range[1])

    dynamics = MountainCarDynamics()
    state1 = (x1, v1)
    next_state = dynamics.step(state1, u1)
    next_next_state = dynamics.step(next_state, u2)
    x2 = next_state[0]; v2 = next_state[1]
    x3 = next_next_state[0]; v3 = next_next_state[1]

    # 2.Obtain the percepts by adding perception error onto the exact state
    perception_error_bound = ( -0.02961, 0.02180)
    perception_error = np.random.uniform(perception_error_bound[0], perception_error_bound[1])
    x1_hat = x1 + perception_error; x2_hat = x2 + perception_error; x3_hat = x3 + perception_error

    ## 3. put the percepts into the MPC
    x0 = np.array([[x1_hat, x2_hat, x3_hat], [v1, v2, v3]])
    error_bound = np.array([[ 0.029, 0.021], [0.001, 0.001]])

    #### Test with three states #####
    #x0 = np.array([-0.05, 0])
    N = 110  # Time horizon
    penalty_start = N - 30
    x_opt, u_opt = Robust_MPC_two_error(x0, error_bound, penalty_start, N)


    ### Then we will test the control actions from MPC.
    # the input will be the x3 and v3
    position_k = x3; velocity_k = v3
    test_pos_list = []; test_vel_list = []
    for k in range(N - 3):
        # Mountain Car dynamics
        force_k = u_opt[k + 2]

        # Apply the Mountain Car dynamics model
        velocity_next = velocity_k + (0.0015 * force_k - 0.0025 * ca.cos(3 * position_k))
        position_next = position_k + velocity_next

        test_pos_list.append(position_next)
        test_vel_list.append(velocity_next)

        velocity_k = velocity_next
        position_k = position_next

    # plt.plot(test_pos_list)
    # plt.title("Position changed with time")
    # plt.xlabel("time step")
    # plt.ylabel("Position Value")
    # plt.show()
    #print(test_pos_list)


    # Check if any value in the list is above 0.45
    if any(pos > 0.45 for pos in test_pos_list):
        # Add the first element of the list to the set
        success_result.add(test_pos_list[0])
    else:
        fail_result.add(test_pos_list[0])

    print(success_result)
    print(fail_result)
    print(fail_result)