import os
import numpy as np
import time
import ray
from algorithms.utils import Config, LogClient, LogServer
from algorithms.algorithm import RL

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

#### general
args.debug = False
args.test = True # if no training, only test
args.profiling = False
backend = 'tensorboard'

#### algorithm and environment
from algorithms.config.SAC import getArgs

import gym
from algorithms.envs.Wrapper import GymWrapper
env_name = 'LunarLander-v2'

args.name='SAC-Lunar-reproduce-high-gamma'

#### misc
args.save_period=900 # in seconds
args.log_period=int(20)
args.seed = None
args.test_interval = int(3e4)
args.n_test = 10

def env_fn():
    return GymWrapper(env_name, 0, 0.2)

agent_args = getArgs(8, 4)

algo_args = Config()
algo_args.replay_size=int(1e6)
algo_args.max_ep_len=1000
algo_args.n_step=int(1e8)
#### checkpoint
algo_args.init_checkpoint = 'checkpoints/SAC-Lunar_LunarLander-v2_SAC_16686/21538671_259.63662453068207.pt' 
algo_args.start_step = 0

agent_args.gamma=0.999

##########################

env = env_fn()
print(f"observation: {env.env.observation_space}, action: {env.env.action_space}")
del env
algo_args.env_fn = env_fn
args.env_fn = env_fn

algo_args.agent_args = agent_args
p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args
if args.debug:
    pi_args.update_interval = 1
    q_args.update_interval = 1
    algo_args.batch_size = 4
    algo_args.max_ep_len=2
    algo_args.replay_size=1
    if not p_args is None:
        p_args.model_buffer_size = 4
    algo_args.n_warmup=1
    args.n_test=1
if args.test:
    algo_args.n_warmup = 0
    args.n_test = 50
    algo_args.n_step = 1
if args.profiling:
    algo_args.batch_size=128
    if algo_args.agent_args.p_args is None:
        algo_args.n_step = 50
    else:
        algo_args.n_step = algo_args.batch_size + 10
        algo_args.replay_size = 1000
        algo_args.n_warmup = algo_args.batch_size
    args.n_test = 1
    algo_args.max_ep_len = 20
if args.seed is None:
    args.seed = int(time.time()*1000)%65536

agent_args.parallel = args.parallel
args.name = f'{args.name}_{env_name}_{agent_args.agent.__name__}_{args.seed}'


if not p_args is None:
    print(f"rollout reuse:{(p_args.refresh_interval/q_args.update_interval*algo_args.batch_size)/p_args.model_buffer_size}")
# each generated data will be used so many times

import torch
torch.set_num_threads(args.n_thread)
print(f"n_threads {torch.get_num_threads()}")
print(f"n_gpus {torch.cuda.device_count()}")

ray.init(ignore_reinit_error = True, num_gpus=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
if args.test or args.debug or args.profiling:
    backend = ''
logger = LogServer.remote({'run_args':args, 'algo_args':algo_args}, backend = backend)
logger = LogClient(logger)
if args.profiling:
    import cProfile
    cProfile.run("RL(logger = logger, run_args=args, **algo_args._toDict()).run()",
                 filename=f'device{args.device}_parallel{args.parallel}.profile')
else:
    RL(logger = logger, run_args=args, **algo_args._toDict()).run()
