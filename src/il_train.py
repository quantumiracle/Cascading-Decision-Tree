# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utils.dataset import Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.heuristic_evaluation import difference_metric, intermediate_features_in_heuristic_tree
from utils.common_func import onehot_coding
from cdt import CDT
from sdt import SDT
import argparse
import json


def train_tree(tree, device, data_path, learner_args):
    criterion = nn.NLLLoss(
    )  # since we already have log probability, simply using Negative Log-likelihood loss can provide cross-entropy loss

    # Load data
    input_path = data_path + 'state_norm.npy'
    label_path = data_path + 'action.npy'
    train_loader = torch.utils.data.DataLoader(
        Dataset(input_path, label_path, partition='train'),
        batch_size=learner_args['batch_size'],
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        Dataset(input_path, label_path, partition='test'),
        batch_size=learner_args['batch_size'],
        shuffle=True,
    )

    # Utility variables
    best_testing_acc = 0.
    testing_acc_list = []

    for epoch in range(1, learner_args['epochs'] + 1):
        epoch_training_loss_list = []
        epoch_feature_difference_list = []

        # Training stage
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target_onehot = onehot_coding(target, device,
                                          learner_args['output_dim'])
            prediction, output, penalty = tree.forward(data)

            difference = 0

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
                    epoch_training_loss_list.append(
                        loss.detach().cpu().data.numpy())
                    print(
                        'Epoch: {:02d} | Batch: {:03d} | CrossEntropy-loss: {:.5f} | Correct: {}/{} | Difference: {}'
                        .format(epoch, batch_idx, loss.data, correct,
                                output.size()[0], difference))

                    tree.save_model(model_path=learner_args['model_path'])

        # intermediate_features = tree.fl_leaf_weights.detach().cpu().numpy()
        # difference = difference_metric(intermediate_features, list2=np.array(intermediate_features_in_heuristic_tree)[:, 1:]) # remove the constants for intermediate feature in heuristic
        # epoch_feature_difference_list.append(difference)

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
        print(
            '\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) | Historical Best: {:.3f}% \n'
            .format(epoch, correct, len(test_loader.dataset), accuracy,
                    best_testing_acc))

    # log data
    # np.save(learner_args['log_path']+'_diff', epoch_feature_difference_list)
    np.save(learner_args['log_path'] + '_acc', testing_acc_list)
    print('Best Testing Accuracy in Training: {}'.format(best_testing_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Imitation learning training.')
    parser.add_argument('--env',
                    dest='EnvName',
                    action='store',
                    default=None)
    parser.add_argument('--method',
                        dest='METHOD',
                        action='store',
                        default=None)

    parser.add_argument('--id', dest='id', action='store', default=0)

    args = parser.parse_args()

    METHOD = args.METHOD  #  one of: 'cdt', 'sdt'

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
        general_il_confs = json.load(
            read_file)  # hyperparameters for rl training

    device = torch.device(il_confs[EnvName]["learner_args"]["device"])

    # add id
    il_confs[EnvName]["learner_args"]["model_path"] = il_confs[EnvName][
        "learner_args"]["model_path"] + args.id
    il_confs[EnvName]["learner_args"][
        "log_path"] = il_confs[EnvName]["learner_args"]["log_path"] + args.id

    if METHOD == 'cdt':
        tree = CDT(il_confs[EnvName]["learner_args"]).to(device)
    elif METHOD == 'sdt':
        tree = SDT(il_confs[EnvName]["learner_args"]).to(device)
    else:
        raise NotImplementedError
    data_path = general_il_confs["data_collect_confs"]["data_path"] + EnvName.split(
        "-")[0].lower() + '/'
    train_tree(tree, device, data_path, il_confs[EnvName]["learner_args"])
