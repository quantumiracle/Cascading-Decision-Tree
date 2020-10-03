import gym
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from rl import PPO
import json
import pickle
from hdt import HeuristicAgentLunarLander

filename = "./il/il.json"
with open(filename, "r") as read_file:
        il_confs = json.load(read_file)  # hyperparameters for rl training

def collect_demo(env, agent, seed=None, render=False, collect_data=False):
    """
    Collect demonstrations.
    """
    env.seed(seed)
    total_reward_list=[]
    a_list=[]
    s_list=[]
    for i in range(il_confs["data_collect_confs"]["episodes"]):
        print('Episode: ', i)
        total_reward = 0
        steps = 0
        s = env.reset()
        while steps < il_confs["data_collect_confs"]["t_horizon"]:
            a, _ = agent.choose_action(s)
            s_list.append(s)
            a_list.append([a])
            s, r, done, info = env.step(a)
            total_reward += r

            if render and not collect_data :
                still_open = env.render()
                if still_open == False: break

            steps += 1
            if done: break

        print("# of episode :{}, reward : {:.1f}, episode length: {}".format(i, total_reward, steps))

        total_reward_list.append(total_reward)  
    print('Average reward: {}'.format(np.mean(total_reward_list)))
    np.save(il_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()+'/state', s_list)
    np.save(il_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()+'/action', a_list)
    return total_reward

def norm_state(env):
    ''' normalize data '''
    file_name="./rl/rl.json"
    with open(file_name, "r") as read_file:
        general_rl_confs = json.load(read_file)  # hyperparameters for rl training
    print(env.spec.id)
    data_path_prefix = general_rl_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()
    with open(data_path_prefix+'/state_info.pkl', 'rb') as f:
        state_stats=pickle.load(f)

    states_data_path = il_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()+'/state'
    states = np.load(states_data_path+'.npy')
    mean = state_stats['mean']
    std = state_stats['std']
    states = (states-mean)/std

    np.save(states_data_path+'_norm', states)
    

if __name__ == '__main__':
    EnvName = 'CartPole-v1'
    # EnvName = 'LunarLander-v2'

    env = gym.make(EnvName)
    if EnvName == 'LunarLander-v2':  # the heuristic agent exists for LunarLander
        agent = HeuristicAgentLunarLander(env, Continuous=False)
    elif EnvName == 'CartPole-v1':  # no heuristic agent for CartPole, so use a well-trained RL agent
        filename = "./mlp/mlp_rl_train.json"
        with open(filename, "r") as read_file:
                rl_confs = json.load(read_file)  # hyperparameters for rl training
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n  # discrete
        agent = PPO(state_dim, action_dim, 'MLP', rl_confs[EnvName]["learner_args"], \
        **rl_confs[EnvName]["alg_confs"]).to(torch.device(rl_confs[EnvName]["learner_args"]["device"]))
        agent.load_model(rl_confs[EnvName]["train_confs"]["model_path"])

    # collect_demo(env, agent, render=False, collect_data = False)
    norm_state(env)

    
    