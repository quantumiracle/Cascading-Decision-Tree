import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import numpy as np
import sys
sys.path.append("..")
from cdt import CDT 
from sdt import SDT 

class PolicyMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyMLP, self).__init__()
        self.fc1   = nn.Linear(state_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, softmax_dim = -1):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, policy_approx = None, learner_args={}, **kwargs):
        super(PPO, self).__init__()
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.lmbda = kwargs['lmbda']
        self.eps_clip = kwargs['eps_clip']
        self.K_epoch = kwargs['K_epoch']
        self.device = torch.device(learner_args['device'])

        hidden_dim = kwargs['hidden_dim']

        self.data = []
        if policy_approx == 'MLP':
            self.policy = PolicyMLP(state_dim, action_dim, hidden_dim).to(self.device)
            self.pi = lambda x: self.policy.forward(x, softmax_dim=-1)
        elif policy_approx == 'SDT':
            self.policy = SDT(learner_args).to(self.device)
            self.pi = lambda x: self.policy.forward(x, LogProb=False)[1]
        elif policy_approx == 'CDT':
            self.policy = CDT(learner_args).to(self.device)
            self.pi = lambda x: self.policy.forward(x, LogProb=False)[1]
        else:
            raise NotImplementedError

        self.fc1   = nn.Linear(state_dim,hidden_dim)
        self.fc_v  = nn.Linear(hidden_dim,1)

        self.optimizer = optim.Adam(list(self.parameters())+list(self.policy.parameters()), lr=self.learning_rate)

    def v(self, x):
        if isinstance(x, (np.ndarray, np.generic) ):
            x = torch.tensor(x)
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.K_epoch):
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach()

            advantage_lst = []
            advantage = 0.0
            for delta_t in torch.flip(delta, [0]):
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def choose_action(self, s, Greedy=False):
        prob = self.pi(torch.from_numpy(s).unsqueeze(0).float().to(self.device)).squeeze()  # make sure input state shape is correct
        if Greedy:
            a = torch.argmax(prob, dim=-1).item()
            return a
        else:
            m = Categorical(prob)
            a = m.sample().item()
            return a, prob  

    def load_model(self, path=None):
        self.load_state_dict(torch.load(path))

