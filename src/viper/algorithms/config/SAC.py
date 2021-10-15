import torch
import ipdb as pdb
import numpy as np
from ..utils import Config, listStack
from ..models import MLP
from ..agents import SAC
import ray
    
def getArgs(observation_dim, n_action):
    """
    hyperparameters refer to the original paper as well as https://stable-baselines3.readthedocs.io/en/master/modules/sac.html 
    """
    p_args=None

    q_args=Config()
    q_args.network = MLP
    q_args.activation=torch.nn.ReLU
    q_args.lr=3e-4
    q_args.sizes = [observation_dim, 16, 32, n_action+1] 
    q_args.update_interval=10
    # MBPO used 1/40 for continous control tasks
    # 1/20 for invert pendulum
    q_args.n_embedding = 0

    pi_args=Config()
    pi_args.network = MLP
    pi_args.activation=torch.nn.ReLU
    pi_args.lr=3e-4
    pi_args.sizes = [observation_dim, 16, 32, n_action] 
    pi_args.update_interval=10

    agent_args=Config()
    agent_args.agent=SAC
    agent_args.eps=0
    agent_args.n_warmup=int(1e3)
    agent_args.batch_size=256 # the same as MBPO
    """
     rainbow said 2e5 samples or 5e4 updates is typical for Qlearning
     bs256lr3e-4, it takes 2e4updates
     for the model on CartPole to learn done...

     Only 3e5 samples are needed for parameterized input continous motion control (refer to MBPO)
     4e5 is needed fore model free CACC (refer to NeurComm)
    """
    agent_args.gamma=0.99
    agent_args.alpha=0.2
    agent_args.target_entropy = 0.2
    # overrides alpha
    # 4 actions, 0.9 greedy = 0.6, 0.95 greedy= 0.37, 0.99 greedy 0.1
    agent_args.target_sync_rate=5e-3
    # called tau in MBPO
    # sync rate per update = update interval/target sync interval
    agent_args.p_args = p_args
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    
    return agent_args

