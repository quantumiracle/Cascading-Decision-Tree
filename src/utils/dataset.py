# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import numpy as np
import json

class Dataset(data.Dataset):
    '''
  Characterizes a dataset for PyTorch
  '''
    def __init__(self,
                 data_path,
                 label_path,
                 partition='all',
                 train_ratio=0.8,
                 total_ratio=1.,
                 ToTensor=True,
                 ):
        """
        Initialization

        :param data_path: (str)
        :param label_path: (str)
        :param partition: (str), choose from all data ('all'), training data ('traing') or testing data ('test')
        :param train_ratio: (float) ratio of training data over all data
        
        """
        self.ToTensor = ToTensor
        # load data
        self.x = np.load(data_path)
        self.y = np.load(label_path)

        total_size = np.array(self.x).shape[0]
        total_size = int(total_size *
                         total_ratio)  # if only use partial dataset
        if partition == 'train':
            self.list_IDs = np.arange(int(total_size * train_ratio))
        elif partition == 'test':
            self.list_IDs = np.arange(int(total_size * train_ratio),
                                      total_size)
        elif partition == 'all':
            self.list_IDs = np.arange(total_size)
        else:
            raise NotImplementedError

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        x = self.x[ID]
        y = self.y[ID]

        if self.ToTensor:
            x = torch.FloatTensor(x)
            # y = torch.FloatTensor(y)
        return x, y
