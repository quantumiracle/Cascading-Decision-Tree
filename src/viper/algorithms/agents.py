from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier as CART
from torch.multiprocessing import Pool, Process, set_start_method
import pickle
from .utils import dictSelect, dictSplit, listStack, parallelEval, sequentialEval, locate
from .models import *
import ipdb as pdb

"""
    Not implemented yet:
        PPO, SAC continous action, MBPO continous action 
    Hierarchiy:
        algorithm
            batchsize
            the number of updates per interaction
            preprocessing the env    
            Both the agent and the model do not care about the tensor shapes or model architecture
        agent
            contains models
            An agent exposes:
                .act() does the sampling
                .update_x(batch), x = p, q, pi
        (abstract) models
            q 
                takes the env, and kwargs
                when continous, = Q(s, a) 
                when discrete, 
                    Q returns Q for all/average/an action,
                    depending on the input action
            pi returns a distribution
        network
            CNN, MLP, ...
"""
class BaseAgent(nn.Module):
    def __init__(self, logger, eps=0, **kwargs):
        super().__init__()
        self.eps = eps
        self.logger = logger
    
    def setEps(self, eps):
        self.eps = eps
        
    def save(self):
        self.logger.save(self)
        
    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])
    
class DecisionTree(BaseAgent):
    def __init__(self, logger, max_depth, max_leaf_nodes, use_Q, **kwargs):
        super().__init__(logger, eps=0)
        self.tree = CART(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        self.use_Q = use_Q
        
    def act(self, s, deterministic=False):
        if isinstance(s, torch.Tensor):
            s = s.numpy()
        p = self.tree.predict_proba(s)
        if not deterministic:
            if random.random()<self.eps:
                return torch.as_tensor(self.action_space.sample())
            else:
                p = torch.tensor(p)
                action = torch.stack([Categorical(item).sample() for item in p])
                return action.cpu().numpy()
        else:
            return p.argmax(axis=1)
        return a
        
    def fit(self, buffer):
        """
        assumes sample weight stored in r
        """
        buffer = buffer.data
        X = np.stack([step['s'] for step in buffer])
        Y = np.stack([step['a'] for step in buffer])
        sample_weight = np.stack([step['r'] for step in buffer])
        self.tree.fit(X, Y, sample_weight)
        correct = self.tree.predict(X) == Y[:, 0]
        self.logger.log(acc =correct.sum()/Y.shape[0])
        
    def save(self):
        self.logger.save(pickle.dumps(self.tree))
        

class QLearning(BaseAgent):
    """ Double Dueling clipped (from TD3) Q Learning"""
    def __init__(self, logger, env, q_args, gamma, eps, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, eps)
        self.gamma = gamma
        self.target_sync_rate=target_sync_rate
        self.action_space=env.action_space

        self.q1 = QCritic(env, **q_args._toDict())
        self.q2 = QCritic(env, **q_args._toDict())
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_args.lr)
        
    def evalQ(self, s, output_distribution, a, **kwargs):
        s, a = locate(self.alpha.device, s, a)
        with torch.no_grad():
            q1 = self.q1(s, output_distribution, a)
            q2 = self.q2(s, output_distribution, a)
            return torch.min(q1, q2)
        
    def updateQ(self, s, a, r, s1, d):
        
        s, a, r, s1, d = locate(self.alpha.device, s, a, r, s1, d)
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            q_next = torch.min(self.q1(s1), self.q2(s1))
            a = q_next.argmax(dim=1)
            q1_pi_targ = self.q1_target(s1, a)
            q2_pi_targ = self.q2_target(s1, a)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        self.logger.log(q=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
        self.q_optimizer.step()

        # Record things
        self.logger.log(q_update=None, q_loss=loss_q/2, reward = r)
        
        # update the target nets
        with torch.no_grad():
            for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                for p, p_targ in zip(current.parameters(), target.parameters()):
                    p_targ.data.mul_(1 - self.target_sync_rate)
                    p_targ.data.add_(self.target_sync_rate * p.data)
                
        
    def act(self, s, deterministic=False):
        """
        s and a are batched
        not differentiable
        """
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s)
        s = s.to(self.alpha.device)
        with torch.no_grad():
            q1 = self.q1(s)
            q2 = self.q2(s)
            q = torch.min(q1, q2)
            a = q.argmax(dim=1)
            if not deterministic and random.random()<self.eps:
                return torch.as_tensor(self.action_space.sample())
            a = a.detach().cpu().numpy()
            return a
        
    def setEps(self, eps):
        self.eps = eps
        
    def save(self):
        self.logger.save(self)
        
    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])

class SAC(QLearning):
    """ Actor Critic (Q function) """
    def __init__(self, logger, env, q_args, pi_args, gamma, target_entropy, target_sync_rate, eps, alpha=0, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env, q_args, gamma, eps, target_sync_rate, **kwargs)
        
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.target_entropy = target_entropy

        self.action_space = env.action_space
        if isinstance(self.action_space, Box): #continous
            self.pi = SquashedGaussianActor(**pi_args._toDict())
        else:
            self.pi = CategoricalActor(**pi_args._toDict())
                                
        if not target_entropy is None:
            pi_params = itertools.chain(self.pi.parameters(), [self.alpha])
        else:
            pi_params = self.pi.parameters()                   
        self.pi_optimizer = Adam(pi_params, lr=pi_args.lr)

    def act(self, s, deterministic=False, output_distribution=False):
        """
            s is batched
            not differentiable
            called during env interaction and model rollout
            not used when updating q
        """
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s)
        s = s.to(self.alpha.device)
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                a = self.pi(s)
                p_a = a
                # [b, n_agent, n_action] or [b, n_action]
                greedy_a = a.argmax(dim=-1)
                stochastic_a = Categorical(a).sample()
                probs = torch.ones(*a.shape)/self.action_space.n
                random_a = Categorical(probs).sample().to(s.device)
                self.logger.log(eps=self.eps)
                if  torch.isnan(a).any():
                    print('action is nan!')
                    a = random_a
                elif deterministic:
                    a = greedy_a
                elif np.random.rand()<self.eps:
                    a = random_a
                else:
                    a = stochastic_a
            else:
                a = self.pi(s, deterministic)
                if isinstance(a, tuple):
                    a = a[0]
            a = a.detach().cpu().numpy()
            if output_distribution:
                return a, p_a.detach()
            else:
                return a
    
    def updatePi(self, s, q = None, **kwargs):
        """
        q is None for single agent
        """
        s = s.to(self.alpha.device)
        if not q is None:
            q = q.to(self.alpha.device)
        if isinstance(self.action_space, Discrete):
            pi = self.pi(s) + 1e-5 # avoid nan
            logp = torch.log(pi/pi.sum(dim=1, keepdim=True))
            if q is None:
                q1 = self.q1(s, output_distribution=True)
                q2 = self.q2(s, output_distribution=True)
                q = torch.min(q1, q2)
            q = q - self.alpha.detach() * logp
            optimum = q.max(dim=1, keepdim=True)[0].detach()
            regret = optimum - (pi*q).sum(dim=1)
            loss = regret.mean()
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            if not self.target_entropy is None:
                alpha_loss = (entropy.detach()-self.target_entropy)*self.alpha
                loss = loss + alpha_loss
            self.logger.log(pi_entropy=entropy, pi_regret=loss, alpha=self.alpha)
        else:
            action, logp = self.pi(s)
            q1 = self.q1(s, action)
            q2 = self.q2(s, action)
            q = torch.min(q1, q2)
            q = q - self.alpha.detach() * logp
            loss = (-q).mean()
            self.logger.log(logp=logp, pi_reward=q)
            
        self.pi_optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.pi.parameters(), max_norm=5, norm_type=2)
            self.pi_optimizer.step()
            if self.alpha < 0:
                self.alpha.data = torch.tensor(0, dtype=torch.float).to(self.alpha.device)

    
    def updateQ(self, s, a, r, s1, d, a1=None, p_a1=None):
        """
            uses alpha to encourage diversity
            for discrete action spaces, different from QLearning since we have a policy network
                takes all possible next actions
            a1 is determinisitc actions of neighborhood,
            only used for decentralized multiagent
            the distribution of local action is recomputed
        """
        s, a, r, s1, d, a1, p_a1 = locate(self.alpha.device, s, a, r, s1, d, a1, p_a1)
        q1 = self.q1(s, False, a)
        q2 = self.q2(s, False, a)
        if a1 is None:
            a1, p_a1 = self.act(s, output_distribution=True)
        
        if isinstance(self.action_space, Discrete):
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                # local a1 distribution
                loga1 = torch.log(p_a1)
                q1_pi_targ = self.q1_target(s1, True, a1) 
                q2_pi_targ = self.q2_target(s1, True, a1)  # [b, n_a]
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha.detach() * loga1
                q_pi_targ = (p_a1*q_pi_targ).sum(dim=1)
                backup = r + self.gamma * (1 - d) * (q_pi_targ)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - backup)**2).mean()
            loss_q2 = ((q2 - backup)**2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            self.logger.log(q=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer.zero_grad()
            if not torch.isnan(loss_q).any():
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
                self.q_optimizer.step()

            # Record things
            self.logger.log(q_update=None, loss_q=loss_q/2, reward = r)

            # update the target nets
            with torch.no_grad():
                for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                    for p, p_targ in zip(current.parameters(), target.parameters()):
                        p_targ.data.mul_(1 - self.target_sync_rate)
                        p_targ.data.add_(self.target_sync_rate * p.data)
        
class MBPO(SAC):
    def __init__(self, env, logger, p_args, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env, **kwargs)
        self.n_p = p_args.n_p
        if isinstance(self.action_space, Box): #continous
            ps = [None for i in range(self.n_p)]
        else:
            ps = [ParameterizedModel(env, logger,**p_args._toDict()) for i in range(self.n_p)]
        self.ps = nn.ModuleList(ps)
        self.p_params = itertools.chain(*[item.parameters() for item in self.ps])
        self.p_optimizer = Adam(self.p_params, lr=p_args.lr)
        
    def updateP(self, s, a, r, s1, d):
        s, a, r, s1, d = locate(self.alpha.device, s, a, r, s1, d)
        loss = 0
        for i in range(self.n_p):
            loss_, s1_ =  self.ps[i](s, a, r, s1, d)
            loss = loss + loss_
        self.p_optimizer.zero_grad()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.p_params, max_norm=5, norm_type=2)
        self.p_optimizer.step()
        return (s1_,)
    
    def roll(self, s, a=None):
        """ batched,
            a is None as long as single agent 
            (if multiagent, set a to prevent computing .act() redundantly)
        """
        s, a = locate(self.alpha.device, s, a)
        p = self.ps[np.random.randint(self.n_p)]
        
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                if a is None:
                    a = self.act(s, deterministic=False)
                r, s1, d = p(s, a)
            else:
                return None
        return  r, s1, d