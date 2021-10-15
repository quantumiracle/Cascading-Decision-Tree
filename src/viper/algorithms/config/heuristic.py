import numpy as np
from ..utils import Config
from ..agents import BaseAgent

def getArgs(env_name):
    agent_args=Config()
    agent_args.p_args=None
    agent_args.q_args=None
    agent_args.pi_args=None
    if env_name == 'CartPole-v1':
        agent_args.agent = lambda **kwargs: HeuristicAgent(cartPoleHeuristic, **kwargs)
    elif env_name == 'LunarLander-v2':
        agent_args.agent = lambda **kwargs: HeuristicAgent(lunarLanderHeuristic, **kwargs)
    return agent_args


class HeuristicAgent(BaseAgent):
    def __init__(self, policy, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

    def act(self, s, deterministic=True):
        assert deterministic==True
        return self.policy(s)
    
def cartPoleHeuristic(observation):
    position, velocity, angle, angle_velocity = observation[0]
    action = int(3. * angle + angle_velocity > 0.)
    return np.array([action])
    
def lunarLanderHeuristic(s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.
    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """
    s = s[0]
    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    return np.array([a])

