import ipdb as pdb
import numpy as np
import torch
import gym
import time
import ray
import random
from tqdm import tqdm, trange
import pickle
from .utils import combined_shape
from .agents import DecisionTree, QLearning

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Utilizes lazy frames of FrameStack to save memory.
    """

    def __init__(self, max_size, action_dtype):
        self.max_size = max_size
        self.data = []
        self.ptr = 0
        self.unread = 0
        self.action_dtype = action_dtype

    def store(self, obs, act, rew, next_obs, done):
        """ not batched """
        if len(self.data) == self.ptr:
            self.data.append({})
        self.data[self.ptr] = {'s':obs, 'a':act, 'r':rew, 's1':next_obs, 'd':done}
        # lazy frames here
        # cuts Q bootstrap if done (next_obs is arbitrary)
        self.ptr = (self.ptr+1) % self.max_size
        
    def storeBatch(self, obs, act, rew, next_obs, done):
        """ 
            explicitly tell if batched since the first dim may be batch or n_agent
            does not convert to tensor, in order to utilze gym FrameStack LazyFrame
        """
        for i in range(done.shape[0]):
            self.store(obs[i], act[i], rew[i], next_obs[i], done[i])

    def sampleBatch(self, batch_size):
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            if key == 'a':
                dtype = self.action_dtype
            else: # done should be float for convenience
                dtype = torch.float
            lst = [torch.as_tensor(dic[key], dtype=dtype) for dic in raw_batch]
            batch[key] = torch.stack(lst)

        return batch
    
    def iterBatch(self, batch_size):
        """ 
        reads backwards from ptr to use the most recent samples,
        not used because it makes learning unstable
        """
        if self.unread == 0:
            return None
        batch_size =  min(batch_size, self.unread)
        read_ptr = self.ptr - (len(self.data) - self.unread)
        idxs = list(range(read_ptr-batch_size, read_ptr))
        idxs = [(i + batch_size*len(self.data))%len(self.data) for i in idxs] 
        # make them in the correct range
        self.unread -= batch_size
        
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            if key == 'a':
                dtype = self.action_dtype
            else: # done should be float for convenience
                dtype = torch.float
            lst = [torch.as_tensor(dic[key], dtype=dtype) for dic in raw_batch]
            batch[key] = torch.stack(lst)
        return batch
    
    def clear(self):
        self.data = []
        self.ptr = 0
        self._rewind()
        
    def _rewind(self):
        self.unread = len(self.data)


class RL(object):
    def __init__(self, logger, run_args, env_fn, agent_args,
        replay_size, start_step,
       max_ep_len,  n_step, init_checkpoint=None,
       **kwargs):
        """ 
        a generic algorithm for single agent model-based actor-critic, 
        can also be used for model-free, actor-free or crtici-free RL
        For MARL, it is better to overload the agent into a meta-agent instead of overloading RL
        warmup:
            model, q, and policy each warmup for n_warmup steps before used
        """
        self.env, self.test_env = env_fn(), env_fn()
        
        agent = agent_args.agent(logger=logger, run_args=run_args, env=self.env, **agent_args._toDict())
        if not init_checkpoint is None:
            with open(init_checkpoint, "rb") as file:
                dic = torch.load(file)
                print(f"loaded {init_checkpoint}")
            agent.load(dic)
            logger.log(interaction=start_step)   
            
        self.name = run_args.name
        self.start_step = start_step
        self.test_interval = run_args.test_interval
        self.n_test = run_args.n_test

        s, self.episode_len, self.episode_reward = self.env.reset(), 0, 0
        self.agent_args = agent_args
        self.p_args, self.pi_args, self.q_args = agent_args.p_args, agent_args.pi_args, agent_args.q_args
        self.agent = agent
        self.n_warmup = agent_args.n_warmup
        # warmup steps before updating pi and q, especially useful for model based agents
        
        self.batch_size = agent_args.batch_size
        self.n_step = n_step
        self.max_ep_len = max_ep_len

        self.logger = logger
        
        # Experience buffer
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dtype = torch.long
        else:
            action_dtype = torch.float
            
        self.env_buffer = ReplayBuffer(max_size=replay_size, action_dtype=action_dtype)
        if not self.p_args is None: # use the model buffer if there is a model
            self.buffer = ReplayBuffer(max_size=self.p_args.model_buffer_size, action_dtype=action_dtype)
        else:
            self.buffer = self.env_buffer  

        # update frequency
        p_args, q_args, pi_args = agent_args.p_args, agent_args.q_args, agent_args.pi_args
        # multiple gradient steps per sample if model based RL
        self.q_update_steps = 1
        self.pi_update_steps = 1
        if not self.p_args is None:
            self.p_update_steps = 1
            self.p_update_steps_warmup = 1
            self.branch = agent_args.p_args.branch
            self.refresh_interval = self.agent_args.p_args.refresh_interval
            self.p_update_interval = p_args.update_interval
            self.p_update_interval_warmup = p_args.update_interval_warmup
            # p may be updated more frequently during warmup, for computational efficiency
            if self.p_update_interval < 1:
                self.p_update_steps = int(1/self.p_update_interval)
                self.p_update_interval = 1
            if self.p_update_interval_warmup < 1:
                self.p_update_steps_warmup = int(1/self.p_update_interval_warmup)
                self.p_update_interval_warmup = 1

        if not self.pi_args is None:
            self.pi_update_interval = pi_args.update_interval
            if self.pi_update_interval < 1:
                self.pi_update_steps = int(1/self.pi_update_interval)
                self.pi_update_interval = 1

        self.q_update_interval = q_args.update_interval
        if self.q_update_interval < 1:
            self.q_update_steps = int(1/self.q_update_interval)
            self.q_update_interval = 1

    def test(self):
        returns = []
        scaled = []
        lengths = []
        episodes = []
        total = 0
        correct = 0
        for i in trange(self.n_test):
            episode = []
            test_env = self.test_env
            test_env.reset()
            d, ep_ret, ep_len = np.array([False]), 0, 0
            while not(d.any() or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                state = torch.as_tensor(test_env.state, dtype=torch.float)
                action = self.agent.act(state.unsqueeze(0), deterministic=True)
                # [b=1, (n_agent), ...]
                if hasattr(self, 'expert'):
                    expert_action = self.expert.act(state.unsqueeze(0), deterministic=True)
                    total += 1
                    correct += (expert_action[0] == action[0])
                _, r, d, _ = test_env.step(action[0])
                episode += [(test_env.state.tolist(), action.tolist(), r.tolist())]
                d=np.array(d)
                ep_ret += r.mean()
                ep_len += 1
            if hasattr(test_env, 'rescaleReward'):
                scaled += [ep_ret]
                ep_ret = test_env.rescaleReward(ep_ret, ep_len)
            returns += [ep_ret]
            lengths += [ep_len]
            episodes += [episode]
        returns = np.stack(returns, axis=0)
        lengths = np.stack(lengths, axis=0)
        self.logger.log(test_episode_reward=returns, test_episode_len=lengths, test_round=None)
        print(self.name)
        if total > 0:
            self.logger.log(imitation_learning_acc = correct/total)
        if hasattr(test_env, 'rescaleReward'):
            self.logger.log(rescaled_reward = np.mean(scaled))
        with open(f"checkpoints/{self.name}/test_result.pickle", "wb") as f:
            pickle.dump(episodes, f)
        with open(f"checkpoints/{self.name}/test_result.txt", "w") as f:
            for episode in episodes:
                for step in episode:
                    f.write(f"{step[0]}, {step[1]}, {step[2]}\n")
                f.write("\n")
        return returns.mean()
        
    def updateAgent(self):
        agent = self.agent
        batch_size = self.batch_size
        env_buffer, buffer = self.env_buffer, self.buffer
        t = self.t
        # Update handling
        if not self.p_args is None:
            if t > self.n_warmup:
                p_update_interval = self.p_update_interval
                p_update_steps = self.p_update_steps
            else:
                p_update_interval = self.p_update_interval_warmup
                p_update_steps = self.p_update_steps_warmup
                
            if (t % p_update_interval) == 0 and t>batch_size:
                for i in range(p_update_steps):
                    batch = env_buffer.sampleBatch(batch_size)
                    agent.updateP(**batch)

        if not self.q_args is None and t>self.n_warmup and t % self.q_update_interval == 0:
            for i in range(self.q_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updateQ(**batch)

        if not self.pi_args is None and t>self.n_warmup and t % self.pi_update_interval == 0:
            for i in range(self.pi_update_steps):
                batch = buffer.sampleBatch(batch_size)
                agent.updatePi(**batch)
                
    def roll(self):
        """
            updates the buffer using model rollouts, using the most recent samples in env_buffer
            stops when the buffer is full (max_size + bacthsize -1) or the env_buffer is exhausted
        """
        env_buffer = self.env_buffer
        buffer = self.buffer
        batch_size = self.batch_size
        env_buffer._rewind()
        buffer.clear()
        batch = env_buffer.sampleBatch(self.batch_size)
        while not batch is None and len(buffer.data) < buffer.max_size:
            s = batch['s']
            a = self.agent.act(s)
            for i in range(self.branch):
                r, s1, d = self.agent.roll(s=s)
                buffer.storeBatch(s, a, r, s1, d)
                if len(buffer.data) >= buffer.max_size:
                    break
            batch = env_buffer.sampleBatch(batch_size)
            
    def step(self):
        env = self.env
        state = env.state
        state = torch.as_tensor(state, dtype=torch.float)
        a = self.agent.act(torch.as_tensor(state, dtype=torch.float).unsqueeze(0))    
        a = a.squeeze(0)
        # Step the env
        s1, r, d, _ = env.step(a)
        self.episode_reward += r
        self.logger.log(interaction=None)
        self.episode_len += 1
        if self.episode_len == self.max_ep_len:
            """
                some envs return done when episode len is maximum,
                this breaks the Markov property
            """
            d = np.zeros(d.shape, dtype=np.float32)
        d = np.array(d)
        self.env_buffer.store(state, a, r, s1, d)
        if d.any() or (self.episode_len == self.max_ep_len):
            """ for compatibility, allow different agents to have different done"""
            self.logger.log(episode_reward=self.episode_reward.mean(), episode_len=self.episode_len, episode=None)
            _, self.episode_reward, self.episode_len = self.env.reset(), 0, 0
        
    def run(self):
        # Main loop: collect experience in env and update/log each epoch
        last_save = 0
        if self.start_step < self.n_warmup and self.pi_args is None: # eps greedy for q learning
            self.agent.setEps(1)
        else:
            self.agent.setEps(self.agent_args.eps)
        pbar = iter(tqdm(range(int(self.n_step)), desc=self.name))
        for t in range(self.start_step, self.n_step): 
            next(pbar)
            self.t = t
            
            if t % self.test_interval == 0:
                mean_return = self.test()
                self.agent.save() # automatically save once per save_period seconds
                self.logger.dump(mean_return) 
                
            self.step()
            
            if not self.p_args is None and t >=self.n_warmup \
                and (t% self.refresh_interval == 0 or len(self.buffer.data) == 0):
                self.roll()
                
            if t == self.n_warmup:
                self.agent.setEps(self.agent_args.eps)
                
            self.updateAgent()        
            
class Dagger(RL):
    def __init__(self, logger, run_args, env_fn, agent_args, expert_args, 
       replay_size, start_step, 
       max_ep_len,  n_step, expert_init_checkpoint=None,
       **kwargs):
        self.env, self.test_env = env_fn(), env_fn()
        self.expert = expert_args.agent(logger=logger, run_args=run_args, env=self.env, **expert_args._toDict())
        self.agent = self.expert
        self.agent_args = agent_args
        self.expert_args = expert_args
        
        if not expert_init_checkpoint is None:
            with open(expert_init_checkpoint, "rb") as file:
                dic = torch.load(file)
            print(f"loaded expert {expert_init_checkpoint}")
            self.expert.load(dic)
            logger.log(interaction=start_step)   
        self.name = run_args.name
        self.start_step = start_step

        s, self.episode_len, self.episode_reward = self.env.reset(), 0, 0

        self.n_step = n_step
        self.max_ep_len = max_ep_len
        self.update_interval = agent_args.update_interval

        self.logger = logger
        self.n_warmup = agent_args.n_warmup
        
        self.test_interval = run_args.test_interval
        self.n_test = run_args.n_test
        
        action_dtype = torch.long
        self.buffer = ReplayBuffer(max_size=replay_size, action_dtype=action_dtype)
        self.q_args = None
        self.p_args = None
        self.pi_args = None
            
    def updateAgent(self):
        if self.t% self.update_interval == 0 and self.t>= self.n_warmup:
            self.agent = DecisionTree(self.logger, **self.agent_args._toDict())
            self.agent.fit(self.buffer)
                
    def step(self):
        env = self.env
        state = env.state
        state = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        # agent
        a = self.agent.act(state)    
        s1, r, d, _ = env.step(a[0])
        # expert
        a = self.expert.act(state)    
        
        self.episode_reward += r
        self.logger.log(interaction=None)
        self.episode_len += 1
        if self.episode_len == self.max_ep_len:
            """
                some envs return done when episode len is maximum,
                this breaks the Markov property
            """
            d = np.zeros(d.shape, dtype=np.float32)
        d = np.array(d)
        if self.agent_args.use_Q:
            q = self.expert.evalQ(state, output_distribution=True, a = None).squeeze(0)
            weight = max(q) - min(q)
        else:
            weight = 1
        self.buffer.store(state.squeeze(0), a, weight, s1, d)
        # stores weight for QDagger
        if d.any() or (self.episode_len == self.max_ep_len):
            """ for compatibility, allow different agents to have different done"""
            self.logger.log(episode_reward=self.episode_reward.mean(), episode_len=self.episode_len, episode=None)
            _, self.episode_reward, self.episode_len = self.env.reset(), 0, 0
