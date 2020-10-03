# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils.dataset import Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.heuristic_evaluation import difference_metric, intermediate_features_in_heuristic_tree
from CDT import CDT
import argparse 

parser = argparse.ArgumentParser(description='parse')
parser.add_argument('--depth1', dest='feature_learning_depth', default=False)
parser.add_argument('--depth2', dest='decision_depth', default=False)
parser.add_argument('--vars', dest='num_intermediate_variables', default=False)
parser.add_argument('--id', dest='id', default=False)
args = parser.parse_args()

def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot

use_cuda = True
learner_args = {
    'num_intermediate_variables': int(args.num_intermediate_variables),
    'feature_learning_depth': int(args.feature_learning_depth),
    'decision_depth': int(args.decision_depth),
    'input_dim': 8,
    'output_dim': 4,
    'lr': 1e-3,
    'weight_decay': 0.,  # 5e-4
    'batch_size': 1280,
    'exp_scheduler_gamma': 1.,
    'cuda': use_cuda,
    'epochs': 40,
    'log_interval': 100,
    'greatest_path_probability': True,
    'beta_fl' : False,  # temperature for feature learning
    'beta_dc' : False,  # temperature for decision making
}
learner_args['model_path'] = './model/cdt/'+str(learner_args['feature_learning_depth'])+'_'+str(learner_args['decision_depth'])+'_var'+str(learner_args['num_intermediate_variables'])+'_id'+str(args.id)


device = torch.device('cuda' if use_cuda else 'cpu')

def train_tree(tree):
    writer = SummaryWriter(log_dir='runs/cdt_'+str(learner_args['feature_learning_depth'])+'_'+str(learner_args['decision_depth'])+'_var'+str(learner_args['num_intermediate_variables'])+'_id'+str(args.id))
    # criterion = nn.CrossEntropyLoss()  # torch CrossEntropyLoss = LogSoftmax + NLLLoss
    criterion = nn.NLLLoss()  # since we already have log probability, simply using Negative Log-likelihood loss can provide cross-entropy loss
        
    # Load data
    data_dir = '../data/discrete_'
    data_path = data_dir+'state.npy'
    label_path = data_dir+'action.npy'
    train_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='train'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(Dataset(data_path, label_path, partition='test'),
                                    batch_size=learner_args['batch_size'],
                                    shuffle=True)


    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []
    
    for epoch in range(1, learner_args['epochs']+1):
        epoch_training_loss_list = []
        epoch_feature_difference_list = []
        
        # Training stage
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target_onehot = onehot_coding(target, device, learner_args['output_dim'])
            prediction, output, penalty = tree.forward(data)

            difference=0
            intermediate_features = tree.fl_leaf_weights.detach().cpu().numpy()
            difference = difference_metric(intermediate_features, list2=np.array(intermediate_features_in_heuristic_tree)[:, 1:]) # remove the constants for intermediate feature in heuristic 
            epoch_feature_difference_list.append(difference)
            
            tree.optimizer.zero_grad()
            loss = criterion(output, target.view(-1))
            loss += penalty
            loss.backward()
            tree.optimizer.step()
            
            # Print intermediate training status
            if batch_idx % learner_args['log_interval'] == 0:
                with torch.no_grad():
                    pred = prediction.data.max(1)[1]
                    correct = pred.eq(target.view(-1).data).sum()
                    loss = criterion(output, target.view(-1))
                    epoch_training_loss_list.append(loss.detach().cpu().data.numpy())
                    print('Epoch: {:02d} | Batch: {:03d} | CrossEntropy-loss: {:.5f} | Correct: {}/{} | Difference: {}'.format(
                            epoch, batch_idx, loss.data, correct, output.size()[0], difference))

                    tree.save_model(model_path = learner_args['model_path'])
        writer.add_scalar('Training Loss', np.mean(epoch_training_loss_list), epoch)
        writer.add_scalar('Training Feature Difference', np.mean(epoch_feature_difference_list), epoch)

        # Testing stage
        tree.eval()
        correct = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size()[0]
            prediction, _, _ = tree.forward(data)
            pred = prediction.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum()
        accuracy = 100. * float(correct) / len(test_loader.dataset)
        if accuracy > best_testing_acc:
            best_testing_acc = accuracy
        testing_acc_list.append(accuracy)
        writer.add_scalar('Testing Accuracy', accuracy, epoch)
        print('\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}% \n'.format(epoch, correct, len(test_loader.dataset), accuracy, best_testing_acc))


if __name__ == '__main__':    
    tree = CDT(learner_args).to(device)
    train_tree(tree)
