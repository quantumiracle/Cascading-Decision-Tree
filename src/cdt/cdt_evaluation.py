# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cdt_plot import draw_tree, get_path
import sys
sys.path.append("..")
from heuristic_evaluation import normalize
import os
import copy

EnvName = 'CartPole-v1'  # LunarLander-v2

__all__ = ['evaluate',
            'plot_importance_single_episode'
            ]

def evaluate(model, tree, episodes=1, frameskip=1, seed=None, DrawTree=None, img_path = 'img/eval_tree'):
    env = gym.make(EnvName)
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    average_weight_list = []

    # show values on tree nodes
    print(tree.state_dict())
    # show probs on tree leaves
    softmax = nn.Softmax(dim=-1)
    print(softmax(tree.state_dict()['dc_leaves']).detach().cpu().numpy())

    for n_epi in range(episodes):
        print('Episode: ', n_epi)
        average_weight_list_epi = []
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while not done:
            a = model(torch.Tensor([s]))
            if step%frameskip==0:
                if DrawTree is not None:
                    draw_tree(tree, input_img=s, DrawTree=DrawTree, savepath=img_path+'_'+DrawTree+'/{:04}.png'.format(step))

            s_prime, r, done, info = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    # np.save('data/s/dt_importance.npy', average_weight_list)

    env.close()


def plot_importance_single_episode(data_path='./data/sdt_importance.npy', save_path='./img/sdt_importance.png', epi_id=0):
    data = np.load(data_path, allow_pickle=True)[epi_id]
    for i, weights_per_feature in enumerate(np.array(data).T):
        plt.plot(weights_per_feature, label='Dim: {}'.format(i))
    plt.legend(loc=4)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    from CDT import Cascade_DDT
    learner_args = {
    'num_intermediate_variables': 2,
    'feature_learning_depth': 2,
    'decision_depth': 2,
    'input_dim': 4,
    'output_dim': 2,
    'lr': 1e-3,
    'weight_decay': 0.,  # 5e-4
    'batch_size': 1280,
    'exp_scheduler_gamma': 1.,
    'cuda': True,
    'epochs': 40,
    'log_interval': 100,
    'greatest_path_probability': True,
    'beta_fl' : False,  # temperature for feature learning
    'beta_dc' : False,  # temperature for decision making
    }
    learner_args['model_path'] = './model/cdt/'+str(learner_args['feature_learning_depth'])+'_'\
        +str(learner_args['decision_depth'])+'_var'+str(learner_args['num_intermediate_variables'])+'_id'+str(4)


    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    learner_args['cuda'] = False  # cpu

    tree = Cascade_DDT(learner_args)
    Discretized=True  # whether load the discretized tree
    if Discretized:
        tree.load_model(learner_args['model_path']+'_discretized')
    else:
        tree.load_model(learner_args['model_path'])

    num_params = 0
    for key, v in tree.state_dict().items():
        print(key, v.reshape(-1).shape[0])
        num_params+=v.reshape(-1).shape[0]
    print('Total number of parameters in model: ', num_params)

    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().numpy()
    img_path = 'img/eval_tree_{}_{}'.format(tree.args['feature_learning_depth'], tree.args['decision_depth'])
    if Discretized:
        img_path += '_discretized'
    evaluate(model, tree, episodes=10, frameskip=1, seed=seed, DrawTree=None, DrawImportance=False, \
        img_path=img_path)

    # plot_importance_single_episode(epi_id=0)