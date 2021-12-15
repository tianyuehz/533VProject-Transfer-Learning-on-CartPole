import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical



class OSI(nn.Module):
    """Network definition to be used for the OSI model"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # NOTE: feel free to experiment with this network

        # self.lin1 = nn.Linear(in_dim, 64)
        # #self.BatchN = nn.BatchNorm1d(64)
        # self.linout = nn.Linear(64, out_dim)
        


        self.lin1 = nn.Linear(in_dim, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.linout = nn.Linear(64, out_dim)

        self.drop = nn.Dropout(p=0.1)

        # initialize weights and bias to 0 in the last layer.
        # this ensures the actors starts out completely random in the beginning, and that the value function starts at 0
        # this can help training.  you can experiment with turning it off.
        # self.linout.bias.data.fill_(0.0)
        # self.linout.weight.data.fill_(0.0)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor):  (BS, in_dim)
        Returns:
            torch.Tensor:  (BS, out_dim)
        """
        #print("inputs", inputs)
        x = self.lin1(inputs)
        #x = self.BatchN(x)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.lin2(x)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.lin3(x)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.linout(x)
        
        return x


class Network(nn.Module):
    """Network definition to be used for actor and critic networks"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # NOTE: feel free to experiment with this network
        self.linin = nn.Linear(in_dim, 200)
        self.linout = nn.Linear(200, out_dim)

        # initialize weights and bias to 0 in the last layer.
        # this ensures the actors starts out completely random in the beginning, and that the value function starts at 0
        # this can help training.  you can experiment with turning it off.
        self.linout.bias.data.fill_(0.0)
        self.linout.weight.data.fill_(0.0)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor):  (BS, in_dim)
        Returns:
            torch.Tensor:  (BS, out_dim)
        """
        #print("inputs", inputs)
        x = self.linin(inputs)
        x = torch.relu(x)
        x = self.linout(x)
        return x




# NOTE: policy gradient methods can handle discrete or continuous actions.
# we include definitions for both cases below.

class DiscreteActor(nn.Module):
    """Actor network that chooses 1 discrete action by sampling from a Categorical distribution of N actions"""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.logits_net = Network(obs_dim, act_dim)

    def forward(self, obs, taken_act=None):
        logits = self.logits_net(obs)
        pi = Categorical(logits=logits)
        logp_a = None
        if taken_act is not None:
            logp_a = pi.log_prob(taken_act)
        return pi, logp_a


class GaussianActor(nn.Module):
    """Actor network that chooses N continuous actions by sampling from N parameterized independent Normal
    distributions """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mu_net = Network(obs_dim, act_dim)
        # make the std learnable, but not dependent on the current observation
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, taken_act=None):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)
        logp_a = None
        if taken_act is not None:
            logp_a = pi.log_prob(taken_act).sum(axis=-1)
        return pi, logp_a


class ActorCritic(nn.Module):
    """
    Object to hold Actor and Critic network objects

    See Sutton book (http://www.incompleteideas.net/book/RLbook2018.pdf) Chapter 13 for discussion of Actor Critic
    methods. Basically they are just policy gradients methods where you also learn a value function and use that to
    aid in learning. Not all options in this class use a critic, for example psi_mode='future_return' just uses the
    rewards in a REINFORCE fashion.
    """

    def __init__(self, obs_dim, act_dim, discrete):
        super().__init__()
        # build actor network
        self.discrete = discrete
        if self.discrete:
            '''uses the same Network to later return prob and logprob of taking an action'''
            self.pi = DiscreteActor(obs_dim, act_dim)
        else:
            '''uses the same Network to later return prob and logprob of taking an action'''
            self.pi = GaussianActor(obs_dim, act_dim)
        # build value function
        self.v = Network(obs_dim, 1)

    def step(self, obs):
        """Run a single forward step of the ActorCritic networks.  Used during rollouts, but not during optimization"""
        # no_grad, since we don't need to do any backprop while we collect data.
        # this means we will have to recompute forward passes later. (this is standard)
        with torch.no_grad():
            '''the prob and logprob of taking an action'''
            #print("obs in AC step", obs.shape)
            pi, _ = self.pi(obs)
            '''sample an action from that prob - we are doing rollouts/inference, so it nots fixed'''
            a = pi.sample()
            '''get its logprob'''
            logp_a = pi.log_prob(a) if self.discrete else pi.log_prob(a).sum(axis=-1)
            '''corresponding Q-value'''
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
