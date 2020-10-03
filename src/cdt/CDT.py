# -*- coding: utf-8 -*-
""" Cascade a feature learning tree and a soft decision tree (sparse in features) """
import torch
import torch.nn as nn
import numpy as np

class CDT(nn.Module):
    def __init__(self, args):
        super(CDT, self).__init__()
        self.args = args
        print('CDT parameters: ', args)
        self.device = torch.device(self.args['device'])

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feature_learning_init()
        self.decision_init()

        self.max_leaf_idx = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args['exp_scheduler_gamma'])

    def get_tree_weights(self, Bias=False):
        """Return tree weights as a list"""
        if Bias:
            return self.state_dict()['fl_inner_nodes.weight'].detach().cpu().numpy(), self.state_dict()['dc_inner_nodes.weight'].detach().cpu().numpy() 
        else:  # no bias
            return self.state_dict()['fl_inner_nodes.weight'][:, 1:].detach().cpu().numpy(), self.state_dict()['dc_inner_nodes.weight'][:, 1:].detach().cpu().numpy() 

    def get_feature_weights(self,):
        return self.state_dict()['fl_leaf_weights'].detach().cpu().numpy().reshape(self.num_fl_leaves, self.args['num_intermediate_variables'], self.args['input_dim'])


    def feature_learning_init(self):
        self.num_fl_inner_nodes = 2**self.args['feature_learning_depth'] -1
        self.num_fl_leaves = self.num_fl_inner_nodes + 1
        self.fl_inner_nodes = nn.Linear(self.args['input_dim']+1, self.num_fl_inner_nodes, bias=False)
        # coefficients of feature combinations
        fl_leaf_weights = torch.randn(self.num_fl_leaves*self.args['num_intermediate_variables'], self.args['input_dim'])
        self.fl_leaf_weights = nn.Parameter(fl_leaf_weights)

        # temperature term
        if self.args['beta_fl'] is True or self.args['beta_fl']==1: # learnable
            beta_fl = torch.randn(self.num_fl_inner_nodes)  # use different beta_fl for each node
            # beta_fl = torch.randn(1)     # or use one beta_fl across all nodes
            self.beta_fl = nn.Parameter(beta_fl)
        elif self.args['beta_fl'] is False or self.args['beta_fl']==0:
            self.beta_fl = torch.ones(1).to(self.device)   # or use one beta_fl across all nodes
        else:  # pass in value for beta_fl
            self.beta_fl = torch.tensor(self.args['beta_fl']).to(self.device)

    def feature_learning_forward(self):
        """ 
        Forward the tree for feature learning.
        Return the probabilities for reaching each leaf.
        """
        path_prob = self.sigmoid(self.beta_fl*self.fl_inner_nodes(self.aug_data))

        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        _mu = self.aug_data.data.new(self.batch_size,1,1).fill_(1.)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.args['feature_learning_depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(self.batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx  # index for each layer
            end_idx = begin_idx + 2 ** (layer_idx+1)
        mu = _mu.view(self.batch_size, self.num_fl_leaves)  

        return mu        


    def decision_init(self):
        self.num_dc_inner_nodes = 2**self.args['decision_depth'] -1
        self.num_dc_leaves = self.num_dc_inner_nodes + 1
        self.dc_inner_nodes = nn.Linear(self.args['num_intermediate_variables']+1, self.num_dc_inner_nodes, bias=False)

        dc_leaves = torch.randn(self.num_dc_leaves, self.args['output_dim'])
        self.dc_leaves = nn.Parameter(dc_leaves)

        # temperature term
        if self.args['beta_dc'] is True or self.args['beta_dc'] == 1: # learnable
            beta_dc = torch.randn(self.num_dc_inner_nodes)  # use different beta_dc for each node
            # beta_dc = torch.randn(1)     # or use one beta_dc across all nodes
            self.beta_dc = nn.Parameter(beta_dc)
        elif self.args['beta_dc'] is False or self.args['beta_dc'] == 0:
            self.beta_dc = torch.ones(1).to(self.device)   # or use one beta_dc across all nodes
        else:  # pass in value for beta_dc
            self.beta_dc = torch.tensor(self.args['beta_dc']).to(self.device)

    def decision_forward(self):
        """
        Forward the differentiable decision tree
        """
        self.intermediate_features_construct()
        aug_features = self._data_augment_(self.features)
        path_prob = self.sigmoid(self.beta_dc*self.dc_inner_nodes(aug_features))
        feature_batch_size = self.features.shape[0]

        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        _mu = aug_features.data.new(feature_batch_size,1,1).fill_(1.)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.args['decision_depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(feature_batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx  # index for each layer
            end_idx = begin_idx + 2 ** (layer_idx+1)
        mu = _mu.view(feature_batch_size, self.num_dc_leaves)  

        return mu   

    def intermediate_features_construct(self):
        """
        Construct the intermediate features for decision making, with learned feature combinations from feature learning module.
        """
        features = self.fl_leaf_weights.view(-1, self.args['input_dim']) @ self.data.transpose(0,1)   # data: (batch_size, feature_dim); return: (num_fl_leaves*num_intermediate_variables, batch)
        self.features = features.contiguous().view(self.num_fl_leaves, self.args['num_intermediate_variables'], -1).permute(2,0,1).contiguous().view(-1, self.args['num_intermediate_variables'])  # return: (N, num_intermediate_variables) where N=batch_size*num_fl_leaves

    def decision_leaves(self, p):
        distribution_per_leaf = self.softmax(self.dc_leaves)
        average_distribution = torch.mm(p, distribution_per_leaf)  # sum(probability of each leaf * leaf distribution)
        return average_distribution

    def forward(self, data, LogProb=True):
        self.data = data
        self.batch_size = data.size()[0]
        self.aug_data = self._data_augment_(data)
        fl_probs = self.feature_learning_forward()  # (batch_size, num_fl_leaves) 
        dc_probs = self.decision_forward()
        dc_probs = dc_probs.view(self.batch_size, self.num_fl_leaves, -1)   # (batch_size, num_fl_leaves, num_dc_leaves)

        _mu = torch.bmm(fl_probs.unsqueeze(1), dc_probs).squeeze(1)  # (batch_size, num_dc_leaves)
        output = self.decision_leaves(_mu)

        if self.args['greatest_path_probability']:
            vs, ids = torch.max(fl_probs, 1)  # ids is the leaf index with maximal path probability
            # get the path with greatest probability, get index of it, feature vector and feature value on that leaf
            self.max_leaf_idx_fl = ids
            self.max_feature_vector = self.fl_leaf_weights.view(self.num_fl_leaves, self.args['num_intermediate_variables'], self.args['input_dim'])[ids]
            self.max_feature_value = self.features.view(-1, self.num_fl_leaves, self.args['num_intermediate_variables'])[:, ids, :]

            one_dc_probs = dc_probs[torch.arange(dc_probs.shape[0]), ids, :]  # select decision path probabilities of learned features with largest probability
            one_hot_path_probability_dc = torch.zeros(one_dc_probs.shape).to(self.device)
            vs_dc, ids_dc = torch.max(one_dc_probs, 1)  # ids is the leaf index with maximal path probability
            self.max_leaf_idx_dc = ids_dc
            one_hot_path_probability_dc.scatter_(1, ids_dc.view(-1,1), 1.)
            prediction = self.decision_leaves(one_hot_path_probability_dc)

        else:  # prediction value equals to the average distribution
            prediction = output

        if LogProb:
            output = torch.log(output)
            prediction = torch.log(prediction)

        return prediction, output, 0
        

    """ Add constant 1 onto the front of each instance, serving as the bias """
    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        input = torch.cat((bias, input), 1)
        return input

    def save_model(self, model_path, id=''):
        torch.save(self.state_dict(), model_path+id)

    def load_model(self, model_path, id=''):
        self.load_state_dict(torch.load(model_path+id, map_location='cpu'))
        self.eval()


if __name__ == '__main__':   
    learner_args = {
        'num_intermediate_variables': 3,
        'feature_learning_depth': 2,
        'decision_depth': 2,
        'input_dim': 8,
        'output_dim': 4,
    }
