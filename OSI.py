import time

import numpy as np
import gym
from gym.spaces import Box, Discrete
#from gym.envs.classic_control.CartPole import 

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from utils import count_vars, discount_cumsum, args_to_str
from models import ActorCritic, OSI
from pg_buffer import PGBuffer, B_Buffer, HBuffer

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import PIL
from models import OSI


def main(args):

    n_mus = 2 #args.n_mus
    act_input = 2
    state_dim = 4
    obs_dim_OSI = state_dim + act_input
    obs_dim_input = state_dim + n_mus

    # Initialize the OSI model  
    #test_input = torch.rand(20, obs_dim_input)

    # Randomly initialize the OSI model
    out_dim = n_mus
    OSI_model = OSI(obs_dim_OSI, out_dim)

    H_dim = 5
    mu_dim = n_mus
    stretch_size = args.K * args.N * args.T

    # Initialize the B buffer
    bufB = B_Buffer(H_dim, mu_dim, stretch_size, args)

    # actor critic 
    discrete = False
    ac = ActorCritic(obs_dim_input, act_input, discrete).to(args.device)

    # Load trained Policy
    model_path = 'saved_policy_models/policy_weight_dec2.pth'
    ac = torch.load(model_path)

    #print("ac model", ac);stop

    for k_index in range(args.K):
        # create environment 
        env = gym.make(args.env)
        env.seed(args.seed)

        print("env gravity, env.masscart, env.masspole, env.length", env.gravity, env.masscart, env.masspole, env.length)#;stop
        #print("observation_space", env.observation_space, env.observation_space.shape)#;stop
        #print("env.action_space", env.action_space)

        new = [env.gravity, env.masscart, env.length][:n_mus]
        use_dyn = True
        env_params = []

        if use_dyn:
            env_params.extend(new)

        for j_index in range(args.N):

            # Prepare for interaction with environment
            start_time = time.time()
            o, ep_ret, ep_len = env.reset(), 0, 0 ; #print("o", o)
            
            # Initialiaze the H buffer
            H = []
            for t_index in range(args.T):

                o = np.concatenate((o, env_params), dtype=np.float32); #print("o", o)
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))
                next_o, r, d, _ = env.step(a)

                #print(next_o.reshape(-1).shape, a)
                #xx = np.append(next_o.reshape(-1), a); print("xx", xx);stop
                H.append(np.append(next_o.reshape(-1), a))  #[next_o, a])
                #H = H + next_o 
                #H.append(a)

                # Update obs (critical!)
                o = next_o

            print("H", len(H)) 
            #o, ep_ret, ep_len = env.reset(), 0, 0 ; #print("o", o)

            for t_index in range(args.T):
                # pops out the oldest
                #print("H before", H)
                pop_o  = H.pop(1)
                #print("pop_o, _", pop_o, _);stop

                #recent obs with action discarded
                recent_o = H[-1][0:4]

                o = np.concatenate((recent_o, env_params), dtype=np.float32); #print("o", o)
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(args.device))
                next_o, r, d, _ = env.step(a)

                # append the newest
                H.append(np.append(next_o.reshape(-1), a))

                # store into H and mu into Buffer B
                bufB.store(H, env_params, args)


    batch = bufB.get()
    #save information
    np.save('save_data/B_buffer_H.npy', batch['H']) # save
    np.save('save_data/B_buffer_mu.npy', batch['mu']) # save
    
    #load information
    #new_num_arr = np.load('data.npy') # load  

    print("bufB", batch['H'].shape, batch['mu'].shape);stop 
 
    # optimize OSI with data in B


    test_input = torch.rand(20, obs_dim_input)

    # Randomly initialize the OSI model
    out_dim = n_mus
    OSI_model = OSI(obs_dim_OSI, out_dim)


    '''
    TODOS:
    - Add terminal
    - Optmize OSI
    - mismatch bad examples
    - connect OSI with UP
    - etc
    '''




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='[CartPole-v0, LunarLander-v2, LunarLanderContinuous-v2, others]')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run')
    parser.add_argument('--K', type=int, default=30, help='Number of K envs to run')
    
    parser.add_argument('--N', type=int, default=5, help='Number of N Hbox')
    parser.add_argument('--T', type=int, default=5, help='Number of pairs in each Hbox')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lam', type=float, default=0.97, help='GAE-lambda factor')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of env steps to run during optimizations')
    parser.add_argument('--max_ep_len', type=int, default=1000)

    parser.add_argument('--train_pi_iters', type=int, default=4)
    parser.add_argument('--train_v_iters', type=int, default=40)
    parser.add_argument('--pi_lr', type=float, default=1e-3, help='Policy learning rate')
    parser.add_argument('--v_lr', type=float, default=3e-4, help='Value learning rate')

    parser.add_argument('--psi_mode', type=str, default='gae',
                        help='value to modulate logp gradient with [future_return, gae]')
    parser.add_argument('--loss_mode', type=str, default='vpg', help='Loss mode [vpg, ppo]')
    parser.add_argument('--clip_ratio', type=float, default=0.1, help='PPO clipping ratio')

    parser.add_argument('--render_interval', type=int, default=100, help='render every N')
    parser.add_argument('--log_interval', type=int, default=100, help='log every N')

    parser.add_argument('--device', type=str, default='cpu', help='you can set this to cuda if you have a GPU')

    parser.add_argument('--suffix', type=str, default='', help='Just for experiment logging (see utils)')
    parser.add_argument('--prefix', type=str, default='logs', help='Just for experiment logging (see utils)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)



    
    

    