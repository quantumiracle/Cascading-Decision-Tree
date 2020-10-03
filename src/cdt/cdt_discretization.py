# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
import sys
import numpy as np
import copy

def discretize_cdt(original_tree, FL=True, DC=True):
    """ 
    Discretize the cascading tree
    if FL: discretize the feature learning tree;  
    if DC: discretize the decision making tree.
    """
    tree = copy.deepcopy(original_tree)
    for name, parameter in tree.named_parameters():

        # discretize feature learning tree and decision making tree separately
        if FL:
            if name == 'beta_fl':
                setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))  # 100 is a large enough value to make soft decision hard

            elif name == 'fl_inner_nodes.weight':
                parameters=[]
                for weights in parameter:
                    bias = weights[0]
                    max_id = np.argmax(np.abs(weights[1:].detach().cpu().numpy()))+1
                    max_v = weights[max_id].detach().cpu().numpy()
                    new_weights = torch.zeros(weights.shape)
                    if max_v>0:
                        new_weights[max_id] = torch.tensor(1)
                    else:
                        new_weights[max_id] = torch.tensor(-1)
                    new_weights[0] = bias/np.abs(max_v)
                    parameters.append(new_weights)

                tree.fl_inner_nodes.weight = nn.Parameter(torch.stack(parameters))

        if DC:
            if name == 'beta_dc':
                setattr(tree, name, nn.Parameter(100*torch.ones(parameter.shape)))

            elif name == 'dc_inner_nodes.weight':
                parameters=[]
                # print(parameter)
                for weights in parameter:
                    bias = weights[0]
                    max_id = np.argmax(np.abs(weights[1:].detach().cpu().numpy()))+1
                    max_v = weights[max_id].detach().cpu().numpy()
                    new_weights = torch.zeros(weights.shape)
                    if max_v>0:
                        new_weights[max_id] = torch.tensor(1)
                    else:
                        new_weights[max_id] = torch.tensor(-1)
                    new_weights[0] = bias/np.abs(max_v)
                    parameters.append(new_weights)

                tree.dc_inner_nodes.weight = nn.Parameter(torch.stack(parameters))

    return tree
