

import casadi as ca
import numpy as np

# Create non-convex optimization problem
opti = ca.Opti()

# Parameters
N = 120  # Time horizon
dt = 0.05  # Time step

# Variables
x = opti.variable(2, N+1)  # State trajectory (position and velocity)
u = opti.variable(1, N)    # Control trajectory (force)
reward = opti.variable()    # Reward variable (negative cost) for reaching the goal

# Adjusted Cost matrices
Q = np.array([[10, 0], [0, 1]])  # Penalty on position and velocity
R = 0.001 * np.eye(1)            # Control cost

# Define the dynamics and cost function
cost = 0
for k in range(N):
    # Cost function: quadratic cost on state and control
    cost += ca.mtimes([x[:, k].T, Q, x[:, k]]) + ca.mtimes([u[:, k].T, R, u[:, k]])

    # Mountain Car dynamics
    position_k = x[0, k]
    velocity_k = x[1, k]
    force_k = u[0, k]

    # Apply the Mountain Car dynamics model
    velocity_next = velocity_k + (0.0015 * force_k - 0.0025 * ca.cos(3 * position_k)) * dt
    position_next = position_k + velocity_next * dt

    # Apply dynamics constraints
    opti.subject_to(x[0, k+1] == position_next)
    opti.subject_to(x[1, k+1] == velocity_next)

# Penalize if goal (position > 0.45) is not reached and add a large reward if it is
reward_value = -1000  # Large reward for reaching the goal
opti.subject_to(reward <= 0)  # Reward should be non-positive (i.e., negative cost)
for k in range(N):
    opti.subject_to(reward <= reward_value * (x[0, k] >= 0.45))  # Give reward if position >= 0.45

# Add the reward to the total cost (reduce cost when reward is triggered)
cost += reward

# Terminal cost to penalize final state deviation
cost += ca.mtimes([x[:, N].T, Q, x[:, N]])

# Set the objective to minimize cost
opti.minimize(cost)

# State constraints
for k in range(N):
    opti.subject_to(x[0, k] >= -1.2)  # Minimum position
    opti.subject_to(x[0, k] <= 0.6)   # Maximum position
    opti.subject_to(x[1, k] >= -0.08)  # Slightly relaxed minimum velocity
    opti.subject_to(x[1, k] <= 0.08)   # Slightly relaxed maximum velocity

    # Adjusted Control input bounds
    opti.subject_to(u[0, k] >= -1.1)
    opti.subject_to(u[0, k] <= 1.1)

# Initial state constraint
x0 = np.array([-0.8, 0.0])  # Example initial state
opti.subject_to(x[:, 0] == x0)

# Set solver (IPOPT for nonlinear problems)
opti.solver('ipopt')

# Solve the optimization problem
sol = opti.solve()

# Extract the solution
x_opt = sol.value(x)
u_opt = sol.value(u)
reward_opt = sol.value(reward)

print("Optimal state trajectory:", x_opt)
print("Optimal control trajectory:", u_opt)
print("Reward (if goal was reached):", reward_opt)








#################
# Create a function to solve the MPC problem for different initial states
def solve_mpc_for_initial_state(x0):
    # Create non-convex optimization problem
    opti = ca.Opti()

    # Parameters
    N = 30  # Horizon length
    dt = 0.05  # Time step

    # Variables
    x = opti.variable(2, N+1)  # State trajectory (position and velocity)
    u = opti.variable(1, N)  # Control trajectory (force)

    # Cost matrices
    Q = 0.001*np.eye(2)  # State cost matrix
    R = 0.01 * np.eye(1)  # Control cost matrix (scaled down)

    # Define the dynamics and cost function
    cost = 0
    for k in range(N):
        # Cost function: quadratic cost on state and control
        cost += ca.mtimes([x[:, k].T, Q, x[:, k]]) + ca.mtimes([u[:, k].T, R, u[:, k]])

        # Mountain Car dynamics:
        position_k = x[0, k]
        velocity_k = x[1, k]
        force_k = u[0, k]

        # Apply the Mountain Car dynamics model
        velocity_next = velocity_k + (0.0015 * force_k - 0.0025 * ca.cos(3 * position_k)) * dt
        position_next = position_k + velocity_next * dt

        # Apply dynamics constraints
        opti.subject_to(x[0, k+1] == position_next)
        opti.subject_to(x[1, k+1] == velocity_next)

    # Terminal cost (penalize final state)
    cost += ca.mtimes([x[:, N].T, Q, x[:, N]])

    # Set the objective to minimize cost
    opti.minimize(cost)

    # State constraints
    for k in range(N):
        opti.subject_to(x[0, k] >= -1.2)  # Minimum position
        opti.subject_to(x[0, k] <= 0.6)   # Maximum position
        opti.subject_to(x[1, k] >= -0.07)  # Minimum velocity
        opti.subject_to(x[1, k] <= 0.07)   # Maximum velocity
        opti.subject_to(u[0, k] >= -1.0)   # Control input bounds
        opti.subject_to(u[0, k] <= 1.0)

    # Goal: reach a position of at least 0.45
    opti.subject_to(x[0, N] >= 0.45)

    # Apply initial state constraint
    opti.subject_to(x[:, 0] == x0)

    # Set solver (IPOPT for nonlinear problems)
    p_opts = {"expand": True}
    s_opts = {"max_iter": 5000, "tol": 1e-6}
    opti.solver('ipopt', p_opts, s_opts)

    # Solve the optimization problem
    sol = opti.solve()

    # Extract the solution
    x_opt = sol.value(x)
    u_opt = sol.value(u)

    # Return the optimal state trajectory and control trajectory
    return x_opt, u_opt

# List of different initial states to test
# initial_states = [
#     np.array([-0.5, 0.03]),
#     np.array([-0.6, 0.02]),
#     np.array([-0.4, 0.01]),
#     np.array([-0.3, 0.05]),
#     np.array([-0.3, 0.03]),
#     np.array([-0.5, 0.00])
# ]

# Define position and velocity ranges
position_range = np.arange(-1, -0.06, 0.02)  # Position from -0.3 to -0.05 with step 0.02
velocity_range = np.arange(-0.07, 0.07, 0.03)  # Velocity from 0.0 to 0.05 with step 0.01

# Generate all combinations of position and velocity
initial_states = [np.array([position, velocity]) for position in position_range for velocity in velocity_range]

# Loop through initial states and solve the MPC problem for each one
success_init_state = []
success_init_action = []
for i, x0 in enumerate(initial_states):
    print(f"\nSolving MPC for initial state {x0}")
    try:
        x_opt, u_opt = solve_mpc_for_initial_state(x0)
        print(f"Optimal state trajectory for x0 = {x0}:\n", x_opt)
        print(f"Optimal control trajectory for x0 = {x0}:\n", u_opt)
        success_init_state.append(x0)
        success_init_action.append(u_opt)

    except Exception as e:
        print(f"Solver failed for initial state {x0}: {e}")

print("here are states reach the points:", success_init_state)