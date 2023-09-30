import gym

class observationFilter(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space["agent"]
        
    def observation(self, observation):
        return observation["agent"]