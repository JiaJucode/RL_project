import os
import gym
import multiprocessing as mp
import tensorflow as tf
from numpy import empty
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from gym_files.wrappers import standStraightReward
from gym_files.wrappers import observationFilter
from tf_agents_local.models.ddpg_buider import DDPG_builder

base_env = gym.make("gym_files/BasicSkateboardEnv-v0", mode = "direct", gravity=10)
gym_env = observationFilter(standStraightReward(base_env))
py_env = gym_wrapper.GymWrapper(gym_env)
env = tf_py_environment.TFPyEnvironment(py_env)

INITIAL_LEARNING_RATE = 3e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=100000,
    decay_rate=0.98)
ACTOR_FC_LAYERS = (64, 64, 32, 16)
CRITIC_FC_LAYERS = (64, 64, 32, 16)

params = [
    [x/10 for x in range(2, 10, 2)], #ou_stddev
    [x/10 for x in range(2, 10, 2)], #ou_damping
    [0.001, 0.01, 0.05], #target_update_tau
    [x/10 for x in range(1, 9, 2)] #gamma
]
builder = DDPG_builder(env = env,
                    checkpoint_dir="tf_agents_files/checkpoint/",
                    log_dir="tf_agents_files/log_files/")

def train_model(std, damping, tau, gamma):
    print("start on:" + str(std) + " " +\
        str(damping) + " " + str(tau) +\
            " " + str(gamma) + "with pid:" + str(os.getpid()))
    params = "std_" + str(std) + "_damping_" + str(damping) +\
            "_tau_" + str(tau) + "_gamma_" + str(gamma)
    checkpoint_dir = builder._checkpoint_dir + params
    log_dir = builder._log_dir + params + "/"
    builder.set_ou_stddev(std).set_ou_damping(damping).\
        set_target_update_tau(tau).set_gamma(gamma).\
        set_checkpoint_dir(checkpoint_dir).set_log_dir(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ddpg_model = builder.build()
    return ddpg_model.train(train_iters=1)

if __name__ == '__main__':
    # TODO search network structure
    ou_stddev = [x/10 for x in range(2, 10, 2)]
    ou_damping = [x/10 for x in range(2, 10, 2)]
    target_update_tau = [0.001, 0.01, 0.05]
    gamma = [x/10 for x in range(1, 9, 2)]
    param_grid = [(std, damping, tau, g)
                  for std in ou_stddev
                  for damping in ou_damping
                  for tau in target_update_tau
                  for g in gamma]
    with mp.Pool(1) as pool:
        results = pool.starmap(train_model, param_grid)
