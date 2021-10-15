import os
import numpy as np
import time
import ray
from algorithms.utils import Config, LogClient, LogServer
from algorithms.algorithm import Dagger
import pdb

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'

    
"""
This section contains run args, separated from args for the RL algorithm and agents
"""
args = Config()
#### computation
os.environ['CUDA_VISIBLE_DEVICES']='1'
args.n_thread = 1
args.parallel = False
args.device = 'cpu'
args.n_cpu = 1 # per agent, used only if parallel = True
args.n_gpu = 0
args.n_run = 3

#### general
args.debug = False
args.test = True # if no training, only test
args.profiling = False
backend = 'tensorboard'

import gym
from algorithms.envs.Wrapper import GymWrapper
env_name = 'CartPole-v1'
#env_name = 'LunarLander-v2'

#args.name='Imitation-QDagger-LunarLander-depth9'
args.name='Imitation-Dagger-CartPole-depth2'

#### misc
args.save_period=99999 # in seconds
args.log_period=int(20)
args.seed = None
args.test_interval = int(3e4)
args.n_test = 50

def env_fn():
    return GymWrapper(env_name, 0, 1)

algo_args = Config()
algo_args.replay_size=int(1e6)
algo_args.max_ep_len=600
algo_args.n_step=int(4)

#### checkpoint
#algo_args.expert_init_checkpoint = 'checkpoints/lunar-1_CartPole-v1_SAC_50644/808241_500.0.pt'
#algo_args.expert_init_checkpoint='checkpoints/SAC-Lunar_LunarLander-v2_SAC_16686/21538671_259.63662453068207.pt' 

algo_args.start_step = 0

#from algorithms.config.SAC import getArgs
#agent_args = getArgs(4, 2)
#agent_args = getArgs(8, 4)
from algorithms.config.heuristic import getArgs
agent_args = getArgs(env_name)
algo_args.expert_args = agent_args
p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args
agent_args.parallel = args.parallel

from algorithms.config.Dagger import getArgs
algo_args.agent_args = getArgs(update_interval=int(1e4), max_depth=2, use_Q=False)

env = env_fn()
print(f"observation: {env.env.observation_space}, action: {env.env.action_space}")
del env
algo_args.env_fn = env_fn
args.env_fn = env_fn

if args.seed is None:
    args.seed = int(time.time()*1000)%65536

if not p_args is None:
    print(f"rollout reuse:{(p_args.refresh_interval/q_args.update_interval*algo_args.batch_size)/p_args.model_buffer_size}")
# each generated data will be used so many times

import torch
torch.set_num_threads(args.n_thread)
print(f"n_threads {torch.get_num_threads()}")
print(f"n_gpus {torch.cuda.device_count()}")

ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))


reward = []
acc = []
short_name = args.name
while args.n_run > 0:
    args.name = f'{short_name}_{env_name}_{agent_args.agent.__name__}_{args.seed}'
    logger = LogServer.remote({'run_args':args, 'algo_args':algo_args}, backend = '')
    logger = LogClient(logger)
    Dagger(logger = logger, run_args=args, **algo_args._toDict()).run()
    reward += [logger.buffer['test_episode_reward']]
    acc += [logger.buffer['imitation_learning_acc']]
    args.n_run -= 1
    args.seed += 1
    
reward = np.stack(reward)
acc = np.stack(acc)
print(np.mean(reward), np.std(reward))
print(np.mean(acc), np.std(acc))