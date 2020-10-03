import gym
import argparse
import matplotlib.pyplot as plt
import numpy as np
from rl.PPO import PPO
import json

filename = "./mlp/mlp_rl_train.json"
with open(filename, "r") as read_file:
        rl_confs = json.load(read_file) 

filename = "./rl/rl.json"
with open(filename, "r") as read_file:
        general_rl_confs = json.load(read_file)  # hyperparameters for rl training

def collect_data(EnvName, learner_args, epi):
    env = gym.make(EnvName)
    print('Env: ', env.spec.id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    model = PPO(state_dim, action_dim, policy_approx='MLP', learner_args=learner_args,  **rl_confs[EnvName]["alg_confs"])
    model.load_model(rl_confs[EnvName]["train_confs"]["model_path"])
    data_path_prefix = general_rl_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()
    a_list=[]
    s_list=[]
    prob_list=[]
    for n_epi in range(epi):
        print('Episode: ', n_epi)
        s = env.reset()
        done = False
        reward = 0.0
        step=0
        while step<1000:
            # a, prob=model.choose_action(s)  # uncomment these if wanna collect output probability rather than action only
            a = model.choose_action(s, Greedy=True)
            s_list.append(s)
            a_list.append([a])
            # prob_list.append(prob.detach().cpu().numpy())
            s, r, done, info = env.step(a)
            step+=1
            if done:
                break
        if n_epi % 100 == 0:      
            np.save(data_path_prefix+'/greedy_state', s_list)
            np.save(data_path_prefix+'/greedy_action', a_list)
            # np.save(env.spec.id+'_ppo_prob', prob_list)
    env.close()


if __name__ == '__main__':
    # EnvName = 'CartPole-v1' 
#     EnvName = 'LunarLander-v2'
    EnvName = 'MountainCar-v0'

    learner_args = {'device': 'cpu'}

    collect_data(EnvName, learner_args, epi=3000)