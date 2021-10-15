import gym
import numpy as np

class GymWrapper(gym.Wrapper):
    """
    Basic wrapper that makes everything a numpy array
    """
    def __init__(self, env, reward_mean=0, reward_std=1):
        gym.Wrapper.__init__(self, gym.make(env))
        self.reward_mean = reward_mean
        self.reward_std = reward_std

    def reset(self):
        state = self.env.reset()
        self.state = np.array(state)
        return self.state
        
    def step(self, a):
        state, reward, done, info = self.env.step(a)
        reward = reward*self.reward_std + self.reward_mean
        self.state = np.array(state)
        return self.state, np.array(reward), np.array(done), None
    
    def rescaleReward(self, sum_reward, ep_len):
        return (sum_reward-ep_len*self.reward_mean)/self.reward_std