import gymnasium as gym
import random
import numpy as np

 
# the action_space_env variable is a list of the actions that the environment
# accepts (e.g., [1] instead of 1)
def dynamic_programming_finite_horizon(MDP, action_space_env, gamma, T):
    ns = len(MDP.keys())
    #print(ns)
    na = len(action_space_env)
    #print(na)
    qs = np.zeros((ns, na, T))
    vs = np.zeros((ns, T))

    t = T-1
    print("Beginning Finite Horizon Dynamic Programming... ")
    # compute q values for each state, action pair
    for s in MDP:
        for ai in range(len(action_space_env)):
            p, r = MDP[s][ai]
            qs[s,ai,t] = np.sum(p*r)
        vs[s,t] = np.max(qs[s,:,t])

    # complete the rest of the table based on the recursion
    while t > 0:
        t -= 1
        for s in MDP:
            for ai in range(len(action_space_env)):
                p, r = MDP[s][ai]
                qs[s,ai,t] = np.sum(p*r)+gamma*np.sum(p*vs[:,t+1])
            vs[s,t] = np.max(qs[s,:,t])
        
    print('Dynamic Programming Complete, Finding Optimal Policy...')
    # then, once complete, compute the policy, which just gets the argmax action q from each state
    policy = dict()
    for s in MDP.keys():
        for t in range(T):
            policy[(s,t)] = np.argmax(qs[s,:,t])
    return policy


def policy_iteration(MDP, action_space_env, gamma):
    policy = dict()
    for s in MDP:
        policy[s] = random.choice(range(len(action_space_env)))
    
    P, R = get_P_R(MDP, policy)
    v = get_v(P, R, gamma)


    improved = True
    while improved:
        improved = False
        for s in MDP:
            #print(s, v[s])
            best_q = v[s]
            best_action = policy[s]
            for ai in range(len(action_space_env)):
                
                p, r = MDP[s][ai]
                q_pi = np.sum(p*r)+gamma*np.sum(p*v)
                if q_pi > best_q:
                    best_q = q_pi
                    best_action = ai
                    improved = True
            if improved:
                policy[s] = best_action
                break
        if improved:

            P, R = get_P_R(MDP, policy)
            v = get_v(P, R, gamma)
                    
    return policy


def value_iteration(MDP, action_space_env, gamma, eps=0.1):
    print("Starting Value Iteration... ")

    # initialize values and constants 
    ns = len(MDP.keys())
    na = len(action_space_env)

    qs = np.zeros((ns, na))
    v_k= -1000*np.ones(ns)

    v_next = np.zeros(ns)
    while True: 

        # compute q values for each state, action pair
        for s in MDP.keys():
            for ai in range(len(action_space_env)):
                p, r = MDP[s][ai]
                qs[s,ai] = np.sum(p*r)+gamma*np.sum(p*v_k)
        v_next = np.max(qs, axis=1)

        # if we didn't improve values, update and return
        if  (np.linalg.norm(v_next-v_k, ord = 1)) < eps:
            v_k = v_next
            break
        v_k = v_next


    # recompuite the q values based on the last values
    for s in MDP.keys():
        for ai in range(len(action_space_env)):
            p, r = MDP[s][ai]
            qs[s,ai] = np.sum(p*r)+gamma*np.sum(p*v_k)
        
    print("Value Iteration Complete, Finding Optimal Policy... ")
    # then, once complete, compute the policy, which just gets the argmax action q from each state
    policy = dict()
    for s in MDP.keys():
        policy[s] = np.argmax(qs[s,:])
    return policy


def get_P_R(MDP, pi):
    num_states = len(MDP.keys())
    # initialize transition matrix and reward vector
    P = np.zeros((num_states,num_states))
    R = np.zeros((num_states))

    num_actions = len(MDP[0].keys())

    for s in MDP:        
        #initialize expected reward and tranisition row
        exp_r = 0
        p = np.zeros(num_states)
        
        # for each action
        ai = pi[s]
        (trans_row, reward) = MDP[s][ai]

        # save row and expected reward
        R[s] = np.sum(reward*trans_row)
        P[s,:] = trans_row
    return (P, R)

def get_v(P, R, gamma):

    # apply the result from solving the bellman equation in matrix form
    return np.linalg.inv(np.eye(P.shape[0])-gamma*P)@R

# obs is a function handle -- if it's not None (as in the case of the pendulum),
# you can call the provided function to get the state from the observation
def get_MDP(env, MDP_state_bounds, action_space, num_data_per_state, obs=None, perception_error_bound=(-0.02961, 0.02180)):

    MDP = dict()

    num_bins = 20

    # maps state number to a nx2 dimensional tuple where n is the number of states and first and second dimensions are lower and upper state bounds.
    state_bounds_dict = get_StateBounds(MDP_state_bounds, num_bins)

    num_states = pow(num_bins,len(MDP_state_bounds))

    print("Building "  +str(num_states) + " State MDP...", end=' ', flush=True)

    for s in range(num_states):
        for a_1 in range(len(action_space)):
            for j in range(num_data_per_state):
                x0_samp = random.uniform(state_bounds_dict[s][0][0], state_bounds_dict[s][0][1])
                v0_samp = random.uniform(state_bounds_dict[s][1][0], state_bounds_dict[s][1][1])
                env.state = np.array([x0_samp, v0_samp])

                observation1, __, done, ___, ____ = env.step([action_space[a_1]])

                observation1[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
                observation1[1] = v0_samp + a_1 * 0.0015 - 0.0025 * np.cos(3 * observation1[0])

                if (observation1[0] < -1.2):
                    observation1[0] = -1.2
                elif (observation1[0] > 0.6):
                    observation1[0] = 0.6
                
                if (observation1[1] < -0.07):
                    observation1[1] = -0.07
                elif (observation1[1] > 0.07):
                    observation1[1] = 0.07

                s1_hat = state_from_obs(observation1, state_bounds_dict)

                act_dict = dict()

                for a_2 in range(len(action_space)):
                    trans_prob = np.zeros(num_states)
                    rewards = np.zeros(num_states)

                    for k in range(num_data_per_state):
                        x1_samp = random.uniform(state_bounds_dict[s1_hat][0][0], state_bounds_dict[s1_hat][0][1])
                        v1_samp = random.uniform(state_bounds_dict[s1_hat][1][0], state_bounds_dict[s1_hat][1][1])
                        env.state = np.array([x1_samp, v1_samp])

                        observation2, reward, done, __, ___ = env.step([action_space[a_2]])

                        observation2[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
                        observation2[1] = v1_samp + a_2 * 0.0015 - 0.0025 * np.cos(3 * observation2[0])

                        if (observation2[0] < -1.2):
                            observation2[0] = -1.2
                        elif (observation2[0] > 0.6):
                            observation2[0] = 0.6
                
                        if (observation2[1] < -0.07):
                            observation2[1] = -0.07
                        elif (observation2[1] > 0.07):
                            observation2[1] = 0.07

                        s2_hat = state_from_obs(observation2, state_bounds_dict)

                        trans_prob[s2_hat] += 1
                        rewards[s2_hat] += reward
                    
                    non_zero_inds = np.where(trans_prob != 0)

                    trans_prob = trans_prob / num_data_per_state
                    rewards[non_zero_inds] = rewards[non_zero_inds] / trans_prob[non_zero_inds]
                    act_dict[a_2] = trans_prob, rewards
                
                MDP[s1_hat] = act_dict
        if (s+1) % 10 == 0: 
            print("Built " + str(s) + '/'+str(num_states) + "   ... ", end = '', flush=True)


    """
    for s in range(num_states):
        # make dict for action to transition probability, reward
        act_dict = dict()
        for a in range(len(action_space)):
            trans_prob = np.zeros(num_states)
            rewards = np.zeros(num_states)

            for j in range(num_data_per_state):
                x_samp = random.uniform(state_bounds_dict[s][0][0], state_bounds_dict[s][0][1])
                v_samp = random.uniform(state_bounds_dict[s][1][0], state_bounds_dict[s][1][1])
                env.state = np.array([x_samp, v_samp])
                
                # get observation, reward
                observation, reward, done, __, ___ = env.step([action_space[a]])
                
                # handle case if observation is not state
                if obs is not None:
                    state_tuple = obs(observation)
                    observation  = np.array(state_tuple)

                """ """
                observation[0] += np.random.uniform(perception_error_bound[0], perception_error_bound[1])
                observation[1] = observation[0] - x_samp

                if (observation[0] < -1.2):
                    observation[0] = -1.2
                elif (observation[0] > 0.6):
                    observation[0] = 0.6
                
                if (observation[1] < -0.07):
                    observation[1] = -0.07
                elif (observation[1] > 0.07):
                    observation[1] = 0.07
                """ """
                    
                new_s = state_from_obs(observation, state_bounds_dict)
                
                trans_prob[new_s] += 1
                rewards[new_s] += reward 
            # trans_prob contains the count of each state visited

            # only average rewards over the states that were visited
            non_zero_inds = np.where(trans_prob != 0)

            trans_prob = trans_prob/num_data_per_state
            rewards[non_zero_inds] = rewards[non_zero_inds] / trans_prob[non_zero_inds]
            act_dict[a] = trans_prob, rewards
        MDP[s] = act_dict
        if (s+1) % 50 == 0: 
            print("Built " + str(s) + '/'+str(num_states) + "   ... ", end = '', flush=True)
    """     



    # return the MDP
    print()
    return MDP


def get_StateBounds(MDP_state_bounds, num_bins):

    # maps state number to a nx2 dimensional tuple where n is the number of states and first and second dimensions are lower and upper state bounds.
    state_bounds_dict = dict()

    state_bound_arrays = []

    for state_bound in MDP_state_bounds:
        boundary_values = np.linspace(state_bound[0], state_bound[1], num=num_bins+1)
        state_bound_arrays.append([[boundary_values[i], boundary_values[i+1]] for i in range(len(boundary_values)-1)])

    # get states as pairwise combinations of all state bound intervals
    all_bound_boxes = []
    cart_prod_recurse(state_bound_arrays, [], all_bound_boxes)

    for i in range(len(all_bound_boxes)): 
        state_bounds_dict[i] = all_bound_boxes[i]
    return state_bounds_dict

def cart_prod_recurse(terms, accum, total):
    last = (len(terms) == 1)
    n = len(terms[0])
    for i in range(n):
        
        item = accum + [terms[0][i]]
        if last:
            total.append(item)
        else:
            cart_prod_recurse(terms[1:], item, total)

def state_from_obs(observation, state_dict):
    for state in state_dict:
        if observation[0] >= state_dict[state][0][0] and observation[0] <= state_dict[state][0][1] and observation[1] >= state_dict[state][1][0] and observation[1] <= state_dict[state][1][1]:
            return state
