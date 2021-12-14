import time

import numpy as np
import gym
from gym.spaces import Box, Discrete
#from gym.envs.classic_control.CartPole import 

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Adadelta

from utils import count_vars, discount_cumsum, args_to_str
from models import ActorCritic, OSI
from pg_buffer import PGBuffer, B_Buffer, HBuffer

from collections import defaultdict
from running_functions import fit, validate, obs_normalizer

from torch.utils.tensorboard import SummaryWriter
import PIL

from models import OSI

# def test_fn()
#     return 42

def main(args):

    n_mus = 2 #args.n_mus
    act_input = 2
    state_dim = 4
    obs_dim_OSI = state_dim + act_input
    obs_dim_input = state_dim + n_mus

    # Load trained OSI
    model_path = 'save_OSI_models/OSI_retrain_mismatch_weight_dec9.pth'
    OSI_estimator = torch.load(model_path)

    out_dim = n_mus

    H_dim = 5
    mu_dim = n_mus
    stretch_size = args.K * args.N * args.T

    # Initialize the B buffer
    #bufB = B_Buffer(H_dim, mu_dim, stretch_size, args)

    # actor critic 
    discrete = False
    ac = ActorCritic(obs_dim_input, act_input, discrete).to(args.device)

    #Load trained Policy
    model_path = 'saved_policy_models/policy_weight_dec2.pth' # Dan model
    ac = torch.load(model_path)

    # create environment 
    env = gym.make(args.env)
    env.seed(args.seed)

    
    true_params = dict(env.gravity= 9.8, env.masscart= 1.0, env.masspole = 0.1, env.length = 0.5)
    
    gravity = np.arange(12.8, 15, 0.1)
    #pole_length = np.arange(0.8, 1.4, 0.1)

    for changing_param in gravity:
        

        print("gravity, env.masscart, env.masspole, env.length", changing_param, env.masscart, env.masspole, env.length)#;stop
        #print("observation_space", env.observation_space, env.observation_space.shape)#;stop
        #print("env.action_space", env.action_space)

         # actually half the pole's length

        new = [changing_param, env.masscart, env.masspole, env.length]
        use_dyn = True
        env_params = []

        if use_dyn:
            env_params.extend(new)

        for j_index in range(args.N):

            # Prepare for interaction with environment
            start_time = time.time()
            o, ep_ret, ep_len = env.reset(), 0, 0 ; #print("o", o)
            
            #print("*********NEW BOX STARTED*********")
            # Initialiaze the H buffer
            H = []
            for t_index in range(args.T):

                o = np.concatenate((o, env_params), dtype=np.float32); #print("o", o)
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))
                next_o, r, d, _ = env.step(a)

                #print("next_o", next_o)

                #print(next_o.reshape(-1).shape, a)
                #xx = np.append(next_o.reshape(-1), a); print("xx", xx);stop
                #H.append(np.append(next_o.reshape(-1) * np.array([10,1,10,1]), a))  #[next_o, a])
                H.append(np.append(next_o.reshape(-1), a))  #[next_o, a])
                #H = H + next_o 
                #H.append(a)

                # Update obs (critical!)
                o = next_o

            '''curious, why do we need this line?'''
            o, ep_ret, ep_len = env.reset(), 0, 0 ; #print("o", o)

            #print("env_params before", env_params)
            for t_index in range(args.T):

                if args.mismatch_OSI:
                    H_input = torch.as_tensor(H, dtype=torch.float32).to(args.device).reshape(1,-1)
                    #print("H_input", H_input.reshape(1,-1).shape)

                    '''using OSI as estimator'''
                    keep_true = env_params
                    out = OSI_estimator(H_input)
                    env_params = out.view(-1).detach().numpy(); #print("env_params", env_params)

                # pops out the oldest
                H.pop(0)
                #recent obs with action discarded
                recent_o = H[-1][0:4]

                o = np.concatenate((recent_o, env_params), dtype=np.float32); #print("o", o)
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))
                next_o, r, d, _ = env.step(a)

                # append the newest
                H.append(np.append(next_o.reshape(-1), a))

                if args.mismatch_OSI:
                    env_params = keep_true

                #print("env_params before", env_params);stop
                # store into H and mu into Buffer B
                #bufB.store(H, env_params, args)