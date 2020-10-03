import sys, math
import numpy as np

import gym
from gym import spaces

class HeuristicAgentLunarLander():
    """
    Heuristic agent for LunarLander environment created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
    """

    def __init__(self, env, Continuous):
        super(HeuristicAgentLunarLander, self).__init__()
        self.continuous = Continuous
    
    def choose_action(self, s,  DIST = False):
        # Heuristic for:
        # 1. Testing. 
        # 2. Demonstration rollout.
        angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
        if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
        if angle_targ < -0.4: angle_targ = -0.4
        hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

        # PID controller: s[4] angle, s[5] angularSpeed
        angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
        #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

        # PID controller: s[1] vertical coordinate s[3] vertical speed
        hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
        #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

        if s[6] or s[7]: # legs have contact
            angle_todo = 0
            hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

        if self.continuous:
            a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
            a = np.clip(a, -1, +1)
        else:
            a = 0  # do nothing
            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2  # fire main
            elif angle_todo < -0.05: a = 3  # fire right
            elif angle_todo > +0.05: a = 1  # fire left
        return a, None

