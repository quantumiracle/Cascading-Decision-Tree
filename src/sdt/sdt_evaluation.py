# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sdt_plot import draw_tree, get_path
import sys
sys.path.append("..")
from heuristic_evaluation import normalize
import os

EnvName = 'CartPole-v1'  # LunarLander-v2
# EnvName = 'LunarLander-v2' 


def evaluate(model, tree, episodes=1, frameskip=1, seed=None, DrawTree=True, DrawImportance=True, WeightedImportance=True, img_path = 'img/eval_tree'):
    env = gym.make(EnvName)
    if seed:
        env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    tree_weights = tree.get_tree_weights()
    average_weight_list = []

    # show values on tree nodes
    print(tree.get_tree_weights(Bias=True))
    # show probs on tree leaves
    softmax = nn.Softmax(dim=-1)
    print(softmax(tree.state_dict()['param']).detach().cpu().numpy())

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
                if DrawTree:
                    draw_tree(tree, (tree.args['input_dim'],), input_img=s, savepath=img_path+'/{:04}.png'.format(step))
                if DrawImportance:
                    path_idx, inner_probs = get_path(tree, s, Probs=True)
                    last_idx=0
                    probs_on_path = []
                    for idx in path_idx[1:]:
                        if idx == 2*last_idx+1:  # parent node goes to left node
                            probs_on_path.append(inner_probs[last_idx])
                        elif idx == 2*last_idx+2:  # last index goes to right node, prob should be 1-prob
                            probs_on_path.append(1-inner_probs[last_idx])
                        else:
                            raise ValueError
                        last_idx = idx
                        
                    weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                    weight_per_node = np.abs(normalize(weights_on_path))
                    if WeightedImportance:  # average weights on path weighted by probabilities
                        weight_per_node = [probs*weights for probs, weights in zip (probs_on_path, weight_per_node)]
                    average_weight = np.mean(weight_per_node, axis=0)  # take absolute to prevent that positive and negative will counteract
                    average_weight_list_epi.append(average_weight)

            s_prime, r, done, info = env.step(a)
            # env.render()
            s = s_prime

            reward += r
            step+=1
            if done:
                break

        average_weight_list.append(average_weight_list_epi)
        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(n_epi, reward, step))
    path = 'data/sdt_importance_online.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/sdt_importance_online.png', epi_id=0)

    env.close()


def evaluate_offline(model, tree, episodes=1, frameskip=1, seed=None, data_path='./data/evaluate_state.npy', DrawImportance=True, method='weight', WeightedImportance=False):
    states = np.load(data_path, allow_pickle=True)
    tree_weights = tree.get_tree_weights()
    average_weight_list=[]
    for n_epi in range(episodes):
        average_weight_list_epi = []
        for i, s in enumerate(states[n_epi]):
            a = model(torch.Tensor([s]))    
            if i%frameskip==0:
                if DrawImportance:
                    if method == 'weight': 
                        path_idx, inner_probs = get_path(tree, s, Probs=True)

                        # get probability on decision path (with greatest leaf probability)
                        last_idx=0
                        probs_on_path = []
                        for idx in path_idx[1:]:
                            if idx == 2*last_idx+1:  # parent node goes to left node
                                probs_on_path.append(inner_probs[last_idx])
                            elif idx == 2*last_idx+2:  # last index goes to right node, prob should be 1-prob
                                probs_on_path.append(1-inner_probs[last_idx])
                            else:
                                raise ValueError
                            last_idx = idx
                            
                        weights_on_path = tree_weights[path_idx[:-1]]  # remove leaf node, i.e. the last index 
                        weight_per_node = np.abs(normalize(weights_on_path))
                        if WeightedImportance:
                            weight_per_node = [probs*weights for probs, weights in zip (probs_on_path, weight_per_node)]
                        average_weight = np.mean(weight_per_node, axis=0)  # take absolute to prevent that positive and negative will counteract
                        average_weight_list_epi.append(average_weight)
                    elif method == 'gradient':
                        x = torch.Tensor([s])
                        x.requires_grad = True
                        a = tree.forward(x)[1] # [1] is output, which requires gradient, but it's the expectation of leaves rather than the max-prob leaf 
                        gradient = torch.autograd.grad(outputs=a, inputs=x, grad_outputs=torch.ones_like(a),
                                            retain_graph=True, allow_unused=True)
                        average_weight_list_epi.append(np.abs(gradient[0].squeeze().cpu().numpy()))

        average_weight_list.append(average_weight_list_epi)
    path = 'data/sdt_importance_offline.npy'
    np.save(path, average_weight_list)
    plot_importance_single_episode(data_path=path, save_path='./img/sdt_importance_offline.png', epi_id=0)

def prediction_evaluation(tree, data_dir='../data/discrete_'):
    from utils.dataset import Dataset
    # Load data
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'

    # a data loader with all data in dataset
    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=int(1e4),
                                    shuffle=True)
    accuracy_list=[]
    correct=0.
    for batch_idx, (data, target) in enumerate(test_loader):
        # target_onehot = onehot_coding(target, tree.args['output_dim'])
        prediction, _, _, _ = tree.forward(data)
        with torch.no_grad():
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
    accuracy = 100. * float(correct) / len(test_loader.dataset)
    print('Tree Accuracy: {:.4f}'.format(accuracy))


def plot_importance_single_episode(data_path='data/sdt_importance.npy', save_path='./img/sdt_importance.png', epi_id=0):
    data = np.load(data_path, allow_pickle=True)[epi_id]
    markers=[".", "d", "o", "*", "^", "v", "p", "h"]
    for i, weights_per_feature in enumerate(np.array(data).T):
        plt.plot(weights_per_feature, label='Dim: {}'.format(i), marker=markers[i], markevery=8)
    plt.legend(loc=1)
    plt.xlabel('Step')
    plt.ylabel('Feature Importance')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Cartpole
    from sdt_train_cartpole import learner_args  # ignore this
    from SDT import SDT

    # for reproduciblility
    seed=3
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    learner_args['cuda'] = False  # cpu
    learner_args['depth'] = 2
    learner_args['model_path'] = './model/sdt/'+str(learner_args['depth'])+'_id'+str(4)

    tree = SDT(learner_args)
    Discretized=False  # whether load the discretized tree
    if Discretized:
        tree.load_model(learner_args['model_path']+'_discretized')
    else:
        tree.load_model(learner_args['model_path'])

    num_params = 0
    for key, v in tree.state_dict().items():
        print(key, v.shape)
        num_params+=v.reshape(-1).shape[0]
    print('Total number of parameters in model: ', num_params)


    model = lambda x: tree.forward(x)[0].data.max(1)[1].squeeze().detach().numpy()
    if Discretized:
        evaluate(model, tree, episodes=10, frameskip=1, seed=seed, DrawTree=False, DrawImportance=False, img_path='img/eval_tree{}_discretized'.format(tree.args['depth']))
    else:
        evaluate(model, tree, episodes=10, frameskip=1, seed=seed, DrawTree=False, DrawImportance=False, img_path='img/eval_tree{}'.format(tree.args['depth']))

    plot_importance_single_episode(epi_id=0)
