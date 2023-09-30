# from gymnasium.envs.registration import register
from gym.envs.registration import register

register(
    id="gym_files/BasicSkateboardEnv-v0",
    entry_point= "gym_files.envs:BasicSkateboardEnv"
)
