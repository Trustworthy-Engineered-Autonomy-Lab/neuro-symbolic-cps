import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gzip
# import torchvision
from torch.utils.data import Subset, TensorDataset

import numpy as np
from PIL import Image, ImageEnhance


##### SET UP Train and Test function with classe for network #####
def test(net, loader, device, batch_size=100):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    
    test_loss = 0
    total=0
    with torch.no_grad():
        for data, target in loader:
            
            # prep data
            data, target = data.to(device), target.to(device)
            #target = torch.reshape(target, (batch_size ,1))
            data = data.float()
            
            # compute floss
            #print(data.shape)
            output = net(data)
            test_loss += F.mse_loss(output, target, reduction='sum')
            
            # update counter for batches
            total = total + 1

    average_loss = test_loss/(len(loader.dataset))
    print('Test set: Avg. loss: {:.12f}'.format(average_loss))
    
    return average_loss.item()

def train(net, loader, optimizer, epoch, device, log_interval=50, batch_size=100):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()
    train_loss = 0
    samples_seen = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        # convert to float and send to device
        target = target.float()  
        data = data.float()
        data, target = data.to(device), target.to(device)
        
        # clear up gradients for backprop
        optimizer.zero_grad()
        output = net(data)

        loss= F.mse_loss(output, target, reduction='sum')
        train_loss += loss

        # compute gradients and make updates
        loss.backward()
        optimizer.step()
    average_loss = train_loss/len(loader.dataset)
    if epoch % log_interval == 0:
        print('\nTrain Epoch: {} Loss: {:.12f}'.format(epoch, average_loss), flush=True)

    return average_loss.item()

    
class Net(nn.Module):
    def __init__(self, num_conv_layers = 3, \
                 input_dim = [400,600], \
                 kernels = [3,4,5], \
                 stride = 2, \
                 conv_in_channels= [3, 100, 100], \
                 conv_out_channels = [100, 100, 100], \
                 pool_size = 2, \
                 pool_stride = 2, \
                 num_lin_layers = 2, \
                 linear_layer_size = 100, \
                 out_size = 1):
        super().__init__()
        layer_list = []
        for i in range(num_conv_layers):
            layer_list.append(nn.Conv2d(in_channels=conv_in_channels[i], 
                                        out_channels=conv_out_channels[i], 
                                        kernel_size=kernels[i], stride=stride))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.MaxPool2d(pool_size, pool_stride))
        layer_list.append(nn.Flatten())
        
        self.feature_extractor = nn.Sequential(*layer_list)

        n_channels = self.feature_extractor(torch.empty(1, conv_in_channels[0], input_dim[0], input_dim[1])).size(-1)
        linear_list = []
        for i in range(num_lin_layers):
            if i == 0:
                linear_list.append( nn.Linear(n_channels, linear_layer_size))
            elif i<num_lin_layers-1:
                linear_list.append( nn.Linear(linear_layer_size, linear_layer_size))
            else:
                linear_list.append( nn.Linear(linear_layer_size, out_size))
            if i == num_lin_layers-1:
                linear_list.append(nn.Tanh())
            else:
                linear_list.append(nn.ReLU())
            
        self.classifier = nn.Sequential(*linear_list)

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return 0.9*out-0.3 # scales and shifts the output to be only in the state space of MC


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

def get_obs_from_render(env):
    obs_im = env.render()
    obs_im = Image.fromarray(obs_im)
    return 1-np.asarray(obs_im.convert('L'))/255.

def get_perturbed_obs_from_render(env, x, alpha, delta):
    env.state = np.array([x,0])
    image = env.render()
    canon_image = Image.fromarray(image)

    enhancer = ImageEnhance.Contrast(canon_image)
    contrast_np = np.asarray(enhancer.enhance(alpha).convert('L'))/255.              
    
    
    env.state = np.array([x-delta, 0])
    image_left = env.render()
    canon_image_left = Image.fromarray(image_left)
    left_enhancer = ImageEnhance.Contrast(canon_image_left)
    canon_image_left_contrast = left_enhancer.enhance(alpha)
    canon_image_np_left = np.asarray(canon_image_left_contrast.convert('L'))/255. #(20x30) np array [0,1]

    env.state = np.array([x+delta, 0])
    image_right = env.render()
    canon_image_right = Image.fromarray(image_right)
    right_enhancer = ImageEnhance.Contrast(canon_image_right)
    canon_image_right_contrast = right_enhancer.enhance(alpha)
    canon_image_np_right = np.asarray(canon_image_right_contrast.convert('L'))/255. #(20x30) np array [0,1]

    level = .5
    blurred_image = (level*canon_image_np_left + contrast_np + level*canon_image_np_right)
    blurred_image = blurred_image/(np.max(blurred_image))
    return 1-blurred_image

def plot_conditions(alphas, deltas, x=.2):

    plt.rcParams["figure.figsize"] = [100.00, 110.50]
    font = {'weight' : 'normal',
            'size'   : 64}

    plt.rc('font', **font)
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env.unwrapped
    env.reset()
    fig, axs = plt.subplots(len(alphas), len(deltas))
    i = 0
    for alpha in alphas:
        j=0
        for delta in deltas:
            alpha_str = str(round(alpha, 2))
            delta_str = str(round(delta, 2))

            image = get_perturbed_obs_from_render(env, x, alpha, delta)
            axs[i,j].imshow(1-image, cmap='gray', vmin=0, vmax=1)
            title = "alpha= " +alpha_str + "\ndelta = " + delta_str
            axs[i,j].set_title(title)
            # Hide X and Y axes label marks
            axs[i,j].xaxis.set_tick_params(labelbottom=False)
            axs[i,j].yaxis.set_tick_params(labelleft=False)

            # Hide X and Y axes tick marks
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            j+=1
        i+=1
    im_name = str("images/TrajConditions_AlphaMin" + str(round(alphas[0],2))+
                    "_AlphaMax"+ str(round(alphas[-1],2))+ 
                    "_DeltaMin" + str(round(deltas[0],2))+
                    "_DeltaMax"+ str(round(deltas[-1],2))+
                    "_x"+str(round(x,2)) +
                    ".png")
    plt.savefig(im_name)  

def parse_yaml_network(model):
    weights = []
    offsets = []

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights.append(np.array(model['weights'][layer]))
        offsets.append(np.array(model['offsets'][layer]))
        if 'Sigmoid' in model['activations'][layer]:
            activations.append(sigmoid)
        elif 'Tanh' in model['activations'][layer]:
            activations.append(np.tanh)
    return (weights, offsets, activations)


def predict_yaml(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 0
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = sigmoid(curNeurons)
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return curNeurons  



def predict(model_tuple, inputs):

    for w,b,sigma in zip(*model_tuple):
        inputs = sigma(w@inputs + b)
    return inputs

def sigmoid(x):

    sigm = 1. / (1. + np.exp(-x))

    return sigm     


if __name__ == '__main__':

    ## set parameters for noise and initial conditions
    nn_controller_filename = "sig16x16.yml"
    discrete_controller_filename = "mc_policy.pkl"

    num_cond = 5
    deltas = np.linspace(0, 0.075, num=num_cond) # Robust Perception trained with 0, 0.075
    alphas = np.linspace(.2, 2, num=num_cond) # Robust Perception trained with .1, 2
    alphas[num_cond//2] = 1 # ensure canonical contrast is included. canonical delta is 0.

    verbose = True # prints a summary of each trajectory attempt
    num_episodes = 10 # number of initial positions to attempt for a given noise parameter set

    # again, these may want to be sampled via .reset() instead of gridded
    x_inits = np.linspace(-0.6, -0.4, num=num_episodes)


    argv = sys.argv
    if len(argv) == 3:
        if argv[1].lower() == "robust":
            perception_type = "robust"
        elif argv[1].lower() == "canonical":
            perception_type = "canonical"
        else:
            print("Please give two argutments <perception_type: \"robust\" or \"canonical\"> <control_type: \"nn\" or \"discrete\"> ")
            exit(1)

        if argv[2].lower() == "nn":
            control_type = "nn"
        elif argv[2].lower() == "discrete":
            control_type = "discrete"
        else:
            print("Please give two argutments <perception_type: \"robust\" or \"canonical\"> <control_type: \"nn\" or \"discrete\"> ")
            exit(1)
    else:
        print("Please give two argutments <perception_type: \"robust\" or \"canonical\"> <control_type: \"nn\" or \"discrete\"> ")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # this should print 'cuda' if you are assigned a GPU
    print(device)

    # just picked a seemingly well-trained examples of each
    if perception_type == "canonical":
        state_dict = torch.load('exampleCanonicalStateEstimator.pth', weights_only=True)
    else:
        state_dict = torch.load('exampleRobustStateEstimator.pth', weights_only=True)
    

    #build network
    state_estimator = Net(  num_conv_layers = 2, \
                    input_dim = [400,600], \
                    kernels = [32,24], \
                    stride = 2, \
                    conv_in_channels= [1, 16], \
                    conv_out_channels = [16, 16], \
                    pool_size = 16, \
                    pool_stride = 2, \
                    num_lin_layers = 2, \
                    linear_layer_size = 100, \
                    out_size = 1)
    
    state_estimator.load_state_dict(state_dict)
    state_estimator.eval()
    state_estimator = state_estimator.to(device)

    np.random.seed(1)

    #num_data_per_state = 50
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array').env.unwrapped
    env_dummy = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    env_dummy = env_dummy.env.unwrapped
    env_dummy.reset()

    pos_min = env.observation_space.low[0]
    pos_max = env.observation_space.high[0]

    vel_min = env.observation_space.low[1]
    vel_max = env.observation_space.high[1]

    act_min = env.action_space.low[0]
    act_max = env.action_space.high[0]

    num_acts = 4
    MDP_state_bounds = [[pos_min, pos_max],[vel_min, vel_max]]
    action_space_env = [act_min+(act_max-act_min)*i/num_acts for i in range(num_acts+1)]

    # might need to change to 110 -----------------------------------------------------------------------------------------------
    T = 110
    
    num_bins = 20 #int(np.sqrt(len(MC_MDP.keys())))


    state_bounds_dict = get_StateBounds(MDP_state_bounds, num_bins)
    # load discrete controller
    with open(discrete_controller_filename, 'rb') as f:
        pi = pickle.load(f)
    
    # load neural network controller
    with open(nn_controller_filename, 'rb') as f:
        nn_controller = yaml.load(f, Loader=yaml.FullLoader)
    
    # pre-loads into .npy arrays for faster forward pass
    nn_controller_tuple = parse_yaml_network(nn_controller)

    # These may want to be sampled instead of gridded. though I grid them here for the display
    print("Generating Example Trajectories...")

    # generate plot to show the environmental conditions
    #plot_conditions(alphas, deltas, x=0.2) # error here <--------------------------------------------------------------------
    all_trajectory_dicts = []


    
    for alpha in alphas:
        for delta in deltas:
            traj_dict = {}
            traj_dict["RealTrajectories"] = []
            traj_dict["EstTrajectories"] = [] 
            traj_dict["Conditions"] = (float(alpha), float(delta))
            # evaluate policy
            all_total_rewards = []
            all_real_pos = []
            all_est_pos = []
            for i in range(num_episodes):

                x_init = x_inits[i]
                env.reset()
                env.state = np.array([x_init, 0])
                
                traj_est = []
                traj_real = []

                total_reward = 0
                all_rewards = []
                
                state_real = env.state
                obs_im = get_perturbed_obs_from_render(env_dummy, state_real[0], alpha, delta)
                max_traj_error = 0 # just to get an idea for how much error was experienced by the controller
                for t in range(T):
                    # batch of 1, channel 1, 400x600 image
                    pos_est = state_estimator(torch.Tensor(obs_im[np.newaxis,np.newaxis,:,:]).to(device))
                    pos_est = pos_est.cpu().detach().numpy()[0][0] # extract float value for position


                    state_est = np.array([pos_est, state_real[1]]) # combine with ground truth velocity

                    # record real and estimated positions
                    traj_est.append(float(pos_est))
                    traj_real.append(float(state_real[0]))

                    abs_error = np.abs(pos_est-state_real[0])
                    if abs_error > max_traj_error:
                        max_traj_error = abs_error

                    if control_type == "nn":
                        action = predict(nn_controller_tuple, state_est)[0]
                    else:
                        # this will need adjustment if time is removed from policy
                        state_ind = state_from_obs(state_est, state_bounds_dict)               
                        action = action_space_env[pi[state_ind]]
                    
                    
                    state_real, reward, done, __, ___ = env.step([action])
                    
                    obs_im = get_perturbed_obs_from_render(env_dummy, state_real[0], alpha, delta)

                    total_reward += reward
                    all_rewards.append(reward)

                    if done or t == T-1:
                        
                        pos_est = state_estimator(torch.Tensor(obs_im[np.newaxis,np.newaxis,:,:]).to(device))
                        pos_est = pos_est.cpu().detach().numpy()[0][0]
                        if verbose:
                            print("alpha: {:.4f}, delta: {:.4f}, initial_x: {:.4f}, max_pos_error : {:.4f}, success?: {}".format(alpha, delta, x_init, max_traj_error, done))
                        state_est = np.array([pos_est, state_real[1]])

                        # record real and estimated positions
                        traj_est.append(float(pos_est))
                        traj_real.append(float(state_real[0]))

                        all_total_rewards.append(total_reward)
                        traj_dict["RealTrajectories"].append(traj_real)
                        traj_dict["EstTrajectories"].append(traj_est)
                        #print("Success!")
                        break

            all_trajectory_dicts.append(traj_dict)
            print("Average Reward over " + str(num_episodes) +" episodes:  " + str(sum(all_total_rewards)/num_episodes), flush=True)
    
            

    traj_filename = str("TrajData_" +perception_type+"_"+control_type+ "_NumCond"+str(num_cond)+
                    "_NumEp"+str(num_episodes)+
                    "_AlphaMin" + str(round(alphas[0],2))+
                    "_AlphaMax"+ str(round(alphas[-1],2))+ 
                    "_DeltaMin" + str(round(deltas[0],2))+
                    "_DeltaMax"+ str(round(deltas[-1],2))+ 
                    ".pkl")
    with open(traj_filename, 'wb') as f:
        pickle.dump(all_trajectory_dicts, f, pickle.HIGHEST_PROTOCOL)
     




