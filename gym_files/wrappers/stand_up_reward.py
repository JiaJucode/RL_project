import math
import gym
import numpy as np


class standStraightReward(gym.Wrapper):
    def __init__(self, env):
        super(standStraightReward, self).__init__(env)
        self.env = env.unwrapped

    def _standing(self) -> bool:
        lf = self.env.get_contact_points("left_foot")
        rf = self.env.get_contact_points("right_foot")
        if (len(lf) == 0 and len(rf) == 0):
            return False
        total = self.env.get_contact_points()
        return total == lf + rf

    def _crawling(self) -> bool:
        lf = self.env.get_contact_points("left_foot")
        rf = self.env.get_contact_points("right_foot")
        lh = self.env.get_contact_points("left_hand")
        rh = self.env.get_contact_points("right_hand")
        
        if (len(lf) == 0 and len(rf) == 0 and len(lh) == 0 and len(rh) == 0):
            return False
        total = self.env.get_contact_points()
        return total == lf + rf + rh + lh

    def _not_lying(self) -> bool:
        head = self.env.get_contact_points("head")
        chest = self.env.get_contact_points("chest")
        hip = self.env.get_contact_points("hip")
        return len(head) + len(chest) + len(hip) == 0

    def _on_ground(self) -> bool:
        return len(self.env.get_contact_points()) > 0

    def step(self, action):
        observation, _, _, info = self.env.step(action)
        # height is the main reward
        reward = 0
        reward += (self.env.get_link_pos("head")[2] \
                 + self.env.get_position()[2] \
                 + self.env.get_link_pos("hip")[2]) / 3 \
                 - (self.env.get_link_pos("right_foot")[2] \
                 + self.env.get_link_pos("left_foot")[2]) / 2

        # higher reward for facing forward or downward
        chest_orientation_y=self.env.get_orientation()[1]
        hip_orientation_y=self.env.get_orientation("hip")[1]
        if (chest_orientation_y >= -math.pi/10 and
            chest_orientation_y <= math.pi/4 and
            hip_orientation_y >= -math.pi/10 and
            hip_orientation_y <= math.pi/4):
            # print("facing forward")
            reward += 0.4
        elif (chest_orientation_y >= math.pi/4 and
              chest_orientation_y <= math.pi*0.7 and
              hip_orientation_y >= math.pi/4 and
              hip_orientation_y <= math.pi*0.7):
            # print("facing downward")
            reward += 0.2
        elif (hip_orientation_y >= -math.pi/1.5 and
              hip_orientation_y <= -math.pi/10):
            # print("facing upward + backward")
            reward -= 0.8

        # big movement = small reward
        reward -= 0.7 * np.mean(np.abs(action)) / self.env.action_space.high[0]

        if self._on_ground():
            if self._standing():
                reward += 0.7
                # print("standing ", reward)
            elif self._crawling():
                reward += 0.5
                # print("standing on four ", reward)
            elif self._not_lying():
                reward += 0.3
                # print("not lying ", reward)
        else:
            if (reward > 0):
                reward = -1.5
            else:
                reward -= 1.5
        return observation, reward, False, info
