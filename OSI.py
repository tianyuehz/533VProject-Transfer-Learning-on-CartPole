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


def main(args):

    n_mus = 2 #args.n_mus
    act_input = 2
    state_dim = 4
    obs_dim_OSI = state_dim + act_input
    obs_dim_input = state_dim + n_mus

    # Load trained OSI
    model_path = 'save_OSI_models/OSI_true_weight_dec9.pth'
    OSI_estimator = torch.load(model_path)

    out_dim = n_mus

    H_dim = 5
    mu_dim = n_mus
    stretch_size = args.K * args.N * args.T

    # Initialize the B buffer
    bufB = B_Buffer(H_dim, mu_dim, stretch_size, args)

    # actor critic 
    discrete = False
    ac = ActorCritic(obs_dim_input, act_input, discrete).to(args.device)

    # Load trained Policy
    #model_path = 'saved_policy_models/policy_weight_dec2.pth' # Dan model
    #ac = torch.load(model_path)

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
                bufB.store(H, env_params, args)


    # batch = bufB.get()
    # #save information
    # np.save('save_data/B_buffer_H_high_var_retrain_mismatch.npy', batch['H']) # save
    # np.save('save_data/B_buffer_mu_high_var_retrain_mismatch.npy', batch['mu']) # save
    
    '''load information'''
    # Part 1 data
    # B_buffer_H = np.load('save_data/B_buffer_H_high_variance.npy'); print("B_buffer_H", B_buffer_H[0,0])
    # B_buffer_mu = np.load('save_data/B_buffer_mu_high_variance.npy'); print("B_buffer_mu", B_buffer_mu[0,0])
    
    # Part 2 Data
    B_buffer_H = np.load('save_data/B_buffer_H_high_var_retrain_mismatch.npy'); #print("B_buffer_H", B_buffer_H[0,0])
    B_buffer_mu = np.load('save_data/B_buffer_mu_high_var_retrain_mismatch.npy'); #print("B_buffer_mu", B_buffer_mu[0,0]);#stop

    # extract the dimensions automatically
    T = B_buffer_H.shape[-2]
    K, N, _, _, _  = B_buffer_H.reshape(30, -1, T, T, 5).shape # KxNxTxTx5


    '''Train-Val Split'''
    #data_ = torch.Tensor(B_buffer_H.reshape(-1, 5))
    #target = torch.Tensor(B_buffer_mu.reshape(-1, 2))

    ''' Shuffle?'''
    B_buffer_H = B_buffer_H.reshape(B_buffer_H.shape[0],-1); #print("B_buffer_H", B_buffer_H.shape, B_buffer_H[0:3])
    # throwout duplicates 
    #print("B_buffer_mu", B_buffer_mu.shape)
    B_buffer_mu = B_buffer_mu[:, 0]; #print("sample B_buffer_mu", B_buffer_mu.shape, B_buffer_mu[380:385]);stop
    
    #seed = 42
    #np.random.seed(seed); np.random.shuffle(B_buffer_H)
    #np.random.seed(seed); np.random.shuffle(B_buffer_mu)

    data_ = torch.Tensor(B_buffer_H)
    target = torch.Tensor(B_buffer_mu); #print(target.shape)

    # avoid overflow
    split_ratio = int(0.8 * data_.shape[0])
    #split_ratio = int(split_ratio//7) * 7; #print("train_ratio", train_ratio)

    data_train, data_val = data_[:split_ratio], data_[split_ratio:]
    target_train, target_val = target[:split_ratio], target[split_ratio:]
    #print("data_train, data_val", data_train.shape, data_val.shape);stop

    #print("bufB", B_buffer_H.shape, B_buffer_mu.shape)
 
    '''optimize OSI with data in B'''
    lr = 1e-5
    out_dim = n_mus
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    '''Optimize OSI'''
    epochs = 500
    batch_size = N * T; #print("batch_size", batch_size);stop

    #n = B_buffer_H.shape[0] * B_buffer_H.shape[1]
    '''Randomly initialize the OSI model'''
    model = OSI(H_dim*T, out_dim).to(device)

    # for re-training
    if args.mismatch_OSI:
        import copy
        model_path = 'save_OSI_models/OSI_true_weight_dec9.pth'
        OSI_estimator = torch.load(model_path)
        #load weights from loaded model
        model.load_state_dict(copy.deepcopy(OSI_estimator.state_dict()))

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() 

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, data_train, target_train, batch_size, optimizer, criterion)
        val_epoch_loss = validate(model, data_val, target_val, batch_size, criterion)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")


    # project_dir = "/home/daniel/Documents/Msc/ubc/UBC Lectures/CPSC 533V/Project/533VProject_modified/"
    # file = project_dir + 'save_OSI_models/OSI_retrain_mismatch_weight_dec9.pth'
    # torch.save(model, file)
    
    '''
    TODOS:
    - Add terminal?
    - Use Helen's updated policy?
    - Shouldnt we fix K=30 envs? No, we sample new mu for every K in pseudocode
    - Ks at OSI and UP? same envs? No, we sample new mu for every K in pseudocode

    1. Train OSI (On true labels), save model
    2. Re-train OSI (On predicted labels using model in 1), save model
    3. Combine 2 with UP to give complete network, save model
    4. Maybe - Refine (policy) in complete network, save model
    5. Train UP-OSI end-to-end (your extension) close to Ben?
    6. Send Michiel an email just in case.


    * Decoupling architecture figure, pseudocode,
    * L1 loss/equations, evaluation range?
    * Metrics- sample efficiency, 
    * Mean and std of the predicted model parameter vs actual/true.
    * Training plots, screenshots etc
    * Ablation studies.
    * terminate rollout? Page 8.
    * We normalize the resulting reward. Page 8.
    (Last implementation/results day).


    - connect OSI with UP as regularizer - Ben__
    '''



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='[CartPole-v0, LunarLander-v2, LunarLanderContinuous-v2, others]')

    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run')
    parser.add_argument('--K', type=int, default=30, help='Number of K envs to run')
    
    parser.add_argument('--N', type=int, default=64, help='Number of N Hbox')
    parser.add_argument('--T', type=int, default=3, help='Number of pairs in each Hbox')
    parser.add_argument('--mismatch_OSI', default=False, help='train/retrain OSI with mismatch')

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



    
    

    