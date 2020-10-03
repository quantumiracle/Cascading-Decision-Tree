""" PPO with cascading decision tree (CDT) as policy function approximator """
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from CDT import CDT 
import sys
sys.path.append("..")
from rl import StateNormWrapper

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
Episodes      = 5000
EnvName = 'CartPole-v1' # LunarLander-v2
# EnvName = 'LunarLander-v2' 
# EnvName = 'FetchReach-v1'

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, learner_args):
        super(PPO, self).__init__()
        self.data = []
        self.model_path = learner_args['model_path']
        self.device = learner_args['device']
        hidden_dim=128
        self.fc1   = nn.Linear(state_dim,hidden_dim)
        # self.fc_pi = nn.Linear(hidden_dim,action_dim)
        self.fc_v  = nn.Linear(hidden_dim,1)

        self.cdt = CDT(learner_args).to(self.device)
        self.pi = lambda x: self.cdt.forward(x, LogProb=False)[1]

        self.optimizer = optim.Adam(list(self.parameters())+list(self.cdt.parameters()), lr=learning_rate)

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

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach()

            advantage_lst = []
            advantage = 0.0
            for delta_t in torch.flip(delta, [0]):
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
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

    def load_model(self, ):
        self.load_state_dict(torch.load(self.model_path))

def run(EnvName, learner_args, train=False, test=False):
    # env = StateNormWrapper(gym.make(EnvName), file_name="../rl/rl_train.json")
    env = gym.make(EnvName)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    print(state_dim, action_dim)
    model = PPO(state_dim, action_dim, learner_args).to(learner_args['device'])
    print_interval = 20
    if test:
        model.load_model()
    rewards_list=[]
    for n_epi in range(Episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done and step < 1000:
            a, prob = model.choose_action(s)
            s_prime, r, done, info = env.step(a)
            if test:
                env.render()
            model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
            # model.put_data((s, a, r, s_prime, prob[a].item(), done))

            s = s_prime

            reward += r
            step+=1
            if done:
                break
        if train:
            model.train_net()
        rewards_list.append(reward)
        if train:   
            if n_epi%print_interval==0 and n_epi!=0:
                # plot(rewards_list)
                np.save(learner_args['log_path'], rewards_list)
                torch.save(model.state_dict(), learner_args['model_path'])
                print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
        else:
            print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))

    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--depth1', dest='feature_learning_depth', default=False)
    parser.add_argument('--depth2', dest='decision_depth', default=False)
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--id', dest='id', default=False)

    args = parser.parse_args()

    env = gym.make(EnvName)
    print('Env info: State space is {}, Action space is {}'.format(env.observation_space, env.action_space))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    env.close()

    learner_args = {
        'num_intermediate_variables': 2,
        'feature_learning_depth': int(args.feature_learning_depth),
        'decision_depth': int(args.decision_depth),
        'input_dim': 4,
        'output_dim': 2,
        'lr': 1e-3,
        'weight_decay': 0.,  # 5e-4
        'batch_size': 1280,
        'exp_scheduler_gamma': 1.,
        'cuda': False,
        'episodes': 40,
        'log_interval': 100,
        'greatest_path_probability': True,
        'beta_fl' : False,  # temperature for feature learning
        'beta_dc' : False,  # temperature for decision making
    }

    file_name=EnvName+'_depth_'+args.feature_learning_depth+'_'+args.decision_depth+'_id'+str(args.id)
    learner_args['model_path'] = '../../data/cdt/model/rl_'+file_name
    learner_args['log_path'] = '../../data/cdt/log/rl_'+file_name
    learner_args['device'] = torch.device('cuda' if learner_args['cuda'] else 'cpu')

    if args.train:
        run(EnvName, learner_args, train=True, test=False)
    if args.test:
        run(EnvName, learner_args, train=False, test=True)
