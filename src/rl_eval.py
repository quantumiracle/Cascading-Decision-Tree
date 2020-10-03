# -*- coding: utf-8 -*-
""" Discretize the (soft) differentiable tree into normal decision tree according to DDT paper"""
import torch
import torch.nn as nn
import os
from utils.dataset import Dataset
import numpy as np
import copy
from utils.common_func import onehot_coding
from cdt import discretize_cdt, CDT
from sdt import discretize_sdt, SDT
from rl import PPO
import json
import argparse
import gym


def evaluation(tree,
               device,
               episodes=100,
               frameskip=1,
               seed=None,
               DrawTree=None,
               img_path=None,
               log_path=None):
    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().cpu().numpy()
    env = gym.make(EnvName)
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    average_weight_list = []
    reward_list = []

    # show values on tree nodes
    # print(tree.state_dict())
    # show probs on tree leaves
    # softmax = nn.Softmax(dim=-1)
    # print(softmax(tree.state_dict()['dc_leaves']).detach().cpu().numpy())

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step = 0
        while not done:
            a = model(torch.Tensor([s]).to(device))
            if step % frameskip == 0:
                if DrawTree is not None:
                    draw_tree(tree,
                              input_img=s,
                              DrawTree=DrawTree,
                              savepath=img_path + '_' + DrawTree +
                              '/{:04}.png'.format(step))

            s_prime, r, done, info = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step += 1
            if done:
                break
        reward_list.append(reward)

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(
            n_epi, reward, step))

        np.save(log_path, reward_list)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reinforcement learning evaluation.')

    parser.add_argument('--env',
                    dest='EnvName',
                    action='store',
                    default=None)

    parser.add_argument('--method',
                        dest='METHOD',
                        action='store',
                        default=None)

    args = parser.parse_args()

    METHOD = args.METHOD  #  one of: 'cdt', 'sdt'

    if METHOD == 'cdt':
        filename = "./cdt/cdt_rl_train.json"
    elif METHOD == 'sdt':
        filename = "./sdt/sdt_rl_train.json"
    else:
        raise NotImplementedError

    EnvName = args.EnvName 

    with open(filename, "r") as read_file:
        rl_confs = json.load(read_file)  # hyperparameters for rl training

    with open('./rl/rl.json', "r") as read_file:
        general_rl_confs = json.load(read_file)  # hyperparameters for rl training

    discretize_type = [True, True]  # for feature learning tree and decision making tree respectively, True means discretization
    device = torch.device('cuda')

    env = gym.make(EnvName)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete

    for idx in range(1, 6):
        # add id
        model_path = rl_confs[EnvName]["train_confs"]["model_path"] + str(idx)
        log_path = rl_confs[EnvName]["train_confs"]["log_path"] + '_eval' + str(idx)
        discretized_log_path = rl_confs[EnvName]["train_confs"]["log_path"] + '_eval_discretized' + str(idx)

        model = PPO(state_dim, action_dim, rl_confs["General"]["policy_approx"], rl_confs[EnvName]["learner_args"], \
        **rl_confs[EnvName]["alg_confs"]).to(device)
        model.load_model(model_path)
        tree = model.policy

        if METHOD == 'cdt':
            discretized_tree = discretize_cdt(tree,
                                              FL=discretize_type[0],
                                              DC=discretize_type[1]).to(device)
        elif METHOD == 'sdt':
            discretized_tree = discretize_sdt(tree).to(device)
        else:
            raise NotImplementedError

        evaluation(tree, device, log_path = log_path, img_path=general_rl_confs["data_collect_confs"]["data_path"]+EnvName.split("-")[0].lower()+'/imgs/')
        evaluation(discretized_tree, device, log_path = discretized_log_path, img_path=general_rl_confs["data_collect_confs"]["data_path"]+EnvName.split("-")[0].lower()+'/imgs/discretized_')

        discretized_tree.save_model(rl_confs[EnvName]["train_confs"]["model_path"] + '_discretized' + str(idx))
