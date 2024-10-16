import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import random


def Pure_MPC_MC(x0, penalty_start, N):
    # Create non-convex optimization problem
    opti = ca.Opti()

    # Variables
    x = opti.variable(2, N + 1)  # State trajectory (position and velocity)
    u = opti.variable(1, N)  # Control trajectory (force)

    # Reward variable (no need to define this as an optimization variable, we will add it directly to the cost)
    reward_value = -1000  # Large reward for reaching the goal (negative cost)

    # Initialize the cost to 0
    cost = 0

    # Define the dynamics and cost function
    for k in range(N):
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
    opti.subject_to(x[:, 0] == x0)

    # Set solver (IPOPT for nonlinear problems)
    opti.solver('ipopt')

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the solution
    x_opt = sol.value(x)
    u_opt = sol.value(u)

    return x_opt, u_opt


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


def Robust_MPC_MC_only_pos(x0, error_bound, penalty_start, N):
    # Create non-convex optimization problem
    pos1 = x0[0,0]; pos2 = x0[0,1]; pos3 = x0[0,2]
    #vel1 = x0[1,0]; vel2 = x0[1,1]; vel3 = x0[1,2]
    pos_lb = error_bound[0, 0]; pos_ub = error_bound[0,1]
    #vel_lb = error_bound[1, 0]; vel_ub = error_bound[1,1]

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
    # vel1_bound = [vel1 - vel_lb, vel1 + vel_ub]
    # vel2_bound = [vel2 - vel_lb, vel2 + vel_ub]
    # vel3_bound = [vel3 - vel_lb, vel3 + vel_ub]


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


def Robust_MPC_two_error_noised_dyn(x0, error_bound, penalty_start, N):
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


    # Define the disturbance bound (e.g., w_k in [-w_max, w_max])
    w_max = 0.001  # Example disturbance bound
    # Define the dynamics and cost function
    for k in range(2, N):
        # Mountain Car dynamics
        position_k = x[0, k]
        velocity_k = x[1, k]
        force_k = u[0, k]

        # Declare the disturbance variable within Opti
        disturbance_k = opti.variable(1)

        # Define disturbance bounds
        opti.subject_to(disturbance_k >= -w_max)
        opti.subject_to(disturbance_k <= w_max)

        # Apply dynamics with disturbance
        velocity_next = velocity_k + (0.0015 * force_k - 0.0025 * ca.cos(3 * position_k)) + disturbance_k
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


#### change the inputs to
x0 = np.array([[-0.5, -0.50168, -0.50502], [0, -0.00168, -0.00334]])
error_bound = np.array([[ 0.029, 0.021], [0.0001, 0.0001]])

#### Test with three states #####
#x0 = np.array([-0.05, 0])
N = 110  # Time horizon
penalty_start = N - 30

x_opt, u_opt = Robust_MPC_two_error_noised_dyn(x0, error_bound, penalty_start, N)
np.savetxt('optimal_action_robust_mpc_noised_dynamic.txt', u_opt, fmt='%.2f')
#### Test the non-robust MPC ####
# test_pos1 = random.uniform(-0.50000 - 0.002, -0.50000 + 0.002)
# test_pos2 = random.uniform(-0.50168 - 0.002, -0.50168 + 0.002)
test_pos3 = random.uniform(-0.50502 - 0.002, -0.50502 + 0.002)

# test_vel1 = 0
# test_vel2 = -0.00168
test_vel3 = -0.00334 # 0.07

position_k = test_pos3; velocity_k = test_vel3
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

plt.plot(test_pos_list)
plt.title("State List Plot")
plt.xlabel("Index")
plt.ylabel("Position Value")
plt.show()
print(test_pos_list)


##### Test a bunch of states ####
#position_range = np.arange(-0.5, -0.4, 0.001)  # Position from -0.3 to -0.05 with step 0.02
#velocity_range = np.arange(-0.04, 0.04, 0.01)  # Velocity from 0.0 to 0.05 with step 0.01
position_range = np.arange(-0.5, -0.4, 0.1)  # Position from -0.3 to -0.05 with step 0.02
velocity_range = np.arange(0, 0.01, 0.01)  # Velocity from 0.0 to 0.05 with step 0.01

# Generate all combinations of position and velocity
# Parameters
N = 110  # Time horizon
penalty_start = N - 30  # Start penalizing from this step
#x0 = np.array([0.2, 0])  # Example initial state
initial_states = [np.array([position, velocity]) for position in position_range for velocity in velocity_range]

# Loop through initial states and solve the MPC problem for each one
success_init_state = []; failed_init_state = []
success_init_action = []; failed_init_action = []
for i, x0 in enumerate(initial_states):
    x_opt, u_opt = Error_bound_MPC_MC(x0, penalty_start, N)
    # print(f"Optimal state trajectory for x0 = {x0}:\n", x_opt)
    # print(f"Optimal control trajectory for x0 = {x0}:\n", u_opt)
    if x_opt[0, -1] > 0.45:
        success_init_state.append(x0)
        success_init_action.append(u_opt)
    else:
        failed_init_state.append(x0)
        failed_init_action.append(u_opt)



pos_list = x_opt[0,:]
#position = [item[0] for item in x_opt]
plt.plot(pos_list)
plt.title("State List Plot")
plt.xlabel("Index")
plt.ylabel("Position Value")
plt.show()