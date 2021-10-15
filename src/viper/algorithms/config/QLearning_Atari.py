import torch
import numpy as np
from ..utils import Config, Logger
from ..models import CNN
from ..agents import QLearning
from ..algorithm import RL
from ..envs.Breakout import env_name, env_fn

"""
    notice that 50M samples is typical for DQNs with visual input (refer to rainbow)
    
    the configs are the same as rainbow,
    batchsize *8, lr * 4, update frequency/ 8
    no noisy q and therefore eps of 3e-2
    
"""
algo_args = Config()

algo_args.max_ep_len=2000
algo_args.batch_size=256
algo_args.n_warmup=int(2e5)
algo_args.replay_size=int(1e6)
# from rainbow
algo_args.test_interval = int(3e4)
algo_args.seed=0
algo_args.save_interval=600
algo_args.log_interval=int(2e2)
algo_args.n_step=int(1e8)

q_args=Config()
q_args.network = CNN
q_args.update_interval=32
q_args.activation=torch.nn.ReLU
q_args.lr=2e-4
q_args.strides = [2]*6
q_args.kernels = [3]*6
q_args.paddings = [1]*6
q_args.sizes = [4, 16, 32, 64, 128, 128, 5] # 4 actions, dueling q learning

agent_args=Config()
agent_args.agent=QLearning
agent_args.eps=3e-2
agent_args.gamma=0.99
agent_args.target_sync_rate=q_args.update_interval/32000

args = Config()
args.env_name="Breakout-v0"
args.name=f"{args.env_name}_{agent_args.agent}"
device = 0

q_args.env_fn = env_fn
agent_args.env_fn = env_fn
algo_args.env_fn = env_fn

agent_args.p_args = None
agent_args.q_args = q_args
agent_args.pi_args = None
algo_args.agent_args = agent_args
args.algo_args = algo_args # do not call toDict() before config is set

RL(logger = Logger(args), device=device, **algo_args._toDict()).run()