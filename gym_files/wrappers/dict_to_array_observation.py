import gym
import numpy as np
from gym.core import Env



class dictToArrayObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        shape = None
        if (self.observation_space.__class__.__name__ == "Dict"):
            for v in self.observation_space.values():
                if shape is None:
                    shape = np.array(v.shape)
                else:
                    shape += np.array(v.shape)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=tuple(shape), dtype=np.float32
            )
        
    def observation(self, observation):
        flattend_obs = []
        for v in observation.values():
            flattend_obs.extend(v)
        return np.array(flattend_obs)
