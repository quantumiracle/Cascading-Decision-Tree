import numpy as np
from gym import spaces
import gym
import json
import pickle

class StateNormWrapper(gym.Wrapper):
    """ 
    Normalize state value for environments.
    """
    def __init__(self, env, file_name):
        super(StateNormWrapper, self).__init__(env)
        with open(file_name, "r") as read_file:
            rl_confs = json.load(read_file)  # hyperparameters for rl training
        print(env.spec.id)
        data_path_prefix = rl_confs["data_collect_confs"]["data_path"]+env.spec.id.split("-")[0].lower()+'/'
        with open(data_path_prefix+'state_info.pkl', 'rb') as f:
            self.state_stats=pickle.load(f)

    def norm(self, s):
        mean =  self.state_stats['mean']
        std = self.state_stats['std']
        s = (s-mean)/std
        return s


    def step(self, a): 
        observation, reward, done, info = self.env.step(a)
        return self.norm(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.norm(observation)

    def render(self, **kwargs):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test
    # EnvName = 'CartPole-v1'
    EnvName = 'LunarLander-v2'

    env = StateNormWrapper(gym.make(EnvName), file_name="rl_train.json")

    for _ in range(10):
        env.reset()
        for _ in range(1000):
            # env.render()
            a = env.action_space.sample()
            s, r, d, _ = env.step(a) # take a random action
            if d:
                break
            print(s)
            # print(s.shape)
    env.close()


