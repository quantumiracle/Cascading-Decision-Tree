# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils.dataset import Dataset
import numpy as np
import copy
from utils.common_func import onehot_coding
from cdt import discretize_cdt, CDT
from sdt import discretize_sdt, SDT
import json
import argparse


def discretization_evaluation(tree, device, discretized_tree, data_path):
    # Load data
    input_path = data_path+'state.npy'
    label_path = data_path+'action.npy'

    # a data loader with all data in dataset
    test_loader = torch.utils.data.DataLoader(Dataset(input_path, label_path, partition='test', ToTensor=True),
                                    batch_size=1280,
                                    shuffle=True)
    accuracy_list=[]
    accuracy_list_=[]
    correct=0.
    correct_=0.
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        target_onehot = onehot_coding(target, device, tree.args['output_dim'])
        prediction, _, _ = tree.forward(data)
        prediction_, _, _ = discretized_tree.forward(data)
        with torch.no_grad():
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
            pred_ = prediction_.data.max(1)[1]
            correct_ += pred_.eq(target.view(-1).data).sum()
    accuracy = 100. * float(correct) / len(test_loader.dataset)
    accuracy_ = 100. * float(correct_) / len(test_loader.dataset)
    print('Original Tree Accuracy: {:.4f} | Discretized Tree Accuracy: {:.4f}'.format(accuracy, accuracy_))
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(
        description='Imitation learning evaluation.')

    parser.add_argument('--env',
            dest='EnvName',
            action='store',
            default=None)
            
    parser.add_argument('--method',
                    dest='METHOD',
                    action='store',
                    default=None) 

    args = parser.parse_args()

    METHOD = args.METHOD     #  one of: 'cdt', 'sdt'

    if METHOD == 'cdt':
        filename = "./cdt/cdt_il_train.json"
    elif METHOD == 'sdt':
        filename = "./sdt/sdt_il_train.json"
    else:
        raise NotImplementedError

    EnvName = args.EnvName 

    with open(filename, "r") as read_file:
        il_confs = json.load(read_file)  # hyperparameters for rl training 
        
    general_filename = "./il/il.json"
    with open(general_filename, "r") as read_file:
        general_il_confs = json.load(read_file)  # hyperparameters for rl training

    discretize_type=[True, True]
    device = torch.device('cuda')

    for idx in range(1,6):
        # add id 
        model_path = il_confs[EnvName]["learner_args"]["model_path"]+str(idx)
        log_path = il_confs[EnvName]["learner_args"]["log_path"]+str(idx)

        if METHOD == 'cdt':
            tree = CDT(il_confs[EnvName]["learner_args"]).to(device)
            tree.load_model(model_path)
            discretized_tree = discretize_cdt(tree, FL=discretize_type[0], DC=discretize_type[1]).to(device)
        elif METHOD == 'sdt':
            tree = SDT(il_confs[EnvName]["learner_args"]).to(device)
            tree.load_model(model_path)
            discretized_tree = discretize_sdt(tree).to(device)
        else:
            raise NotImplementedError

        data_path = general_il_confs["data_collect_confs"]["data_path"]+EnvName.split("-")[0].lower()+'/'
        discretization_evaluation(tree, device, discretized_tree, data_path)

        discretized_tree.save_model(model_path = model_path+'_discretized')

