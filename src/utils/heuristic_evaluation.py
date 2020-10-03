# -*- coding: utf-8 -*-
""" This script contains information for heuristic decision tree of LunarLander-v2 (discrete case) """
import torch
import torch.nn as nn
import numpy as np

# In non-hierarchical heuristic decision tree, weights (weights&bias) of all nodes are listed here 
nodes_in_heuristic_tree = [  
    [0, 0,0,0,0,0,0,1,1],  # first dim is bias, the rest are weights

    [-0.4, 0.5, 0,1,0,0,0,0,0],
    [-0.4, -0.5, 0,-1,0,0,0,0,0],
    [0, 1,0,0,0,0,0,0,0],

    # at
    [0.2, 0,0,0,0,-0.5,-1,0,0],
    [0.15, 0,0,0,0,-0.5,-1,0,0],
    [-0.25, 0,0,0,0,0.5,1,0,0],

    [-0.2, 0,0,0,0,-0.5,-1,0,0],
    [-0.25, 0,0,0,0,-0.5,-1,0,0],
    [0.15, 0,0,0,0,0.5,1,0,0],


    [0, 0.25, 0, 0.5, 0, -0.5, -1, 0, 0 ],
    [-0.05, 0.25, 0, 0.5, 0, -0.5, -1, 0, 0 ],
    [-0.05, -0.25, 0, -0.5, 0, 0.5, 1, 0, 0 ],


    # ht cases
    [-0.05, 0.275, -0.5, 0, -0.5, 0,0,0,0],

    [-0.05, -0.275, -0.5, 0, -0.5, 0,0,0,0],

    [-0.05, 0, 0, 0, -0.5, 0, 0, 0, 0],

    # at, ht cases
    [-0.2, 0.275, -0.5, 0,-0.5, 0.5,1,0,0],
    [0.2, 0.275, -0.5, 0,-0.5, -0.5, -1, 0,0],

    [-0.2, -0.275, -0.5, 0,-0.5, 0.5,1,0,0],
    [0.2, -0.275, -0.5, 0,-0.5, -0.5, -1, 0,0],

    [0.2, 0.275, -0.5, 0,-0.5, 0.5,1,0,0],
    [-0.2, 0.275, -0.5, 0,-0.5, -0.5, -1, 0,0],

    [0.2, -0.275, -0.5, 0,-0.5, 0.5,1,0,0],
    [-0.2, -0.275, -0.5, 0,-0.5, -0.5, -1, 0,0],

    [0, 0.025, -0.5, -0.5, -0.5, 0.5, 1, 0, 0],
    [0, 0.525, -0.5, 0.5, -0.5, -0.5, -1, 0, 0],

    [0, -0.525, -0.5, -0.5, -0.5, 0.5, 1, 0, 0],
    [0, -0.025, -0.5, 0.5, -0.5, -0.5, -1, 0, 0],

]

# All intermediate feature vectors in heuristic decision tree
intermediate_features_in_heuristic_tree = [ # first dim is constant, the rest are weights
    # at
    [0, 0,0,0,0,0,0,0,0],
    [0.2, 0,0,0,0,-0.5,-1,0,0],
    [-0.2, 0,0,0,0,-0.5,-1,0,0],
    [0, 0.25,0,0.5,0,-0.5,-1,0,0],
    
    # ht
    [0, 0,0,0,-0.5,0,0,0,0],
    [0, 0.275,-0.5,0,-0.5,0,0,0,0],
    [0, 0.275,-0.5,0,-0.5,0,0,0,0],
]

def normalize(list_v):
    normalized_list = []
    for v in list_v:
        if np.sum(np.abs(v)) == 0:
            continue
        else:
            v =np.array(v)/np.sum(np.abs(v))
        normalized_list.append(v)
    return normalized_list

def l1_norm(a,b):
    return np.linalg.norm(np.array(a)-np.array(b), ord=1)

def difference_metric(list1, list2=nodes_in_heuristic_tree, norm=True):
    '''
    Calculate minimal difference of list1 and list2
    '''
    if norm:
        list1 = normalize(list1)
        list2 = normalize(list2)
    score = []
    for v1 in list1:
        sim_list = []
        for v2 in list2:
            sim = np.min([l1_norm(v1, v2),  l1_norm(v1, -1.*np.array(v2))])
            sim_list.append(sim)
        score.append(np.min(sim_list))  # should be changed to be mean rather than sum
    return np.mean(score)

if __name__ == '__main__':  
    a=np.ones((2,8))
    print(difference_metric(a, nodes_in_heuristic_tree))
