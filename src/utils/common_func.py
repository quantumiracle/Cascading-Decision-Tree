import matplotlib.pyplot as plt
import torch

def plot(rewards, name:str):
    # clear_output(True)
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.savefig(name)
    # plt.show()
    plt.clf()  
    plt.close()


import json
def json_read_file(filename):
    '''
    @brief:
        read data from json file
    @params:
        filename
    @return:
        (dict) parsed json file
    '''
    with open(filename, "r") as read_file:
        return json.load(read_file)


def onehot_coding(target, device, output_dim):
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.)
    return target_onehot