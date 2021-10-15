import torch
import ipdb as pdb
import numpy as np
from ..agents import DecisionTree
from ..utils import Config
import ray

    
def getArgs(target_acc=None, max_depth=20, max_leaf_nodes=int(1e6), update_interval=1000, use_Q=False):
    # target acc overides max_depth and max_leaf_nodes

    p_args=None
    q_args=None
    pi_args=None

    agent_args=Config()
    agent_args.agent=DecisionTree
    agent_args.p_args = p_args
    agent_args.eps = 0
    agent_args.q_args = q_args
    agent_args.pi_args = pi_args
    agent_args.max_depth=max_depth
    agent_args.max_leaf_nodes=max_leaf_nodes
    agent_args.use_Q = use_Q
    agent_args.n_warmup = update_interval
    agent_args.update_interval = update_interval
    return agent_args

