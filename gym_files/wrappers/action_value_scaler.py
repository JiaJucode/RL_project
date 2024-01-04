import gym
import numpy as np
from gym.core import Env

class actionValueScaler(gym.ActionWrapper):
    def __init__(self, env: Env, low = -1, high = 1):
        super().__init__(env)
        self.scale_factor = (self.action_space.high - self.action_space.low) / (high - low)
        self.old_low = self.action_space.low
        self.new_low = low
        self.env.action_space = gym.spaces.Box(low, high, shape=self.action_space.shape, dtype=np.float32)
        
    def action(self, action):
        return (np.array(action) - self.new_low) * self.scale_factor + self.old_low