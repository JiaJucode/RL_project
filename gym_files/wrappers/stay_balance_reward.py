import gym
import numpy as np

class stayBalanceReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env.unwrapped
        self.counter = 0

    def _standing(self) -> bool:
        lf = len(self.env.get_contact_points("left_foot"))
        rf = len(self.env.get_contact_points("right_foot"))
        if (lf == 0 and rf == 0):
            return False
        total = len(self.env.get_contact_points())
        return total == lf + rf

    def _on_ground(self) -> bool:
        return len(self.env.get_contact_points()) > 0

    # def step(self, action):
    #     observation, _, _, info = self.env.step(action)
    #     self.counter += 1
    #     reward = (self.env.get_link_pos("head")[2] \
    #              + self.env.get_position()[2] \
    #              + self.env.get_link_pos("hip")[2]) / 3 \
    #              - (self.env.get_link_pos("right_foot")[2] \
    #              + self.env.get_link_pos("left_foot")[2]) / 2

    #     # # big movement = small reward
    #     # reward -= 0.7 * np.mean(np.abs(action)) / self.env.action_space.high[0]
    #     if self._standing():
    #         return observation, reward, False, info
    #     if not self._on_ground():
    #         return observation, 0, False, info
    #     return observation, -1, True, info

    def step(self, action):
        observation, _, _, info = self.env.step(action)
        return observation, -np.mean(np.abs(action)) / self.env.action_space.high[0], False, info

    def reset(self):
        self.counter = 0
        return self.env.reset()

    