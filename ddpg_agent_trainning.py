import gym
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from gym_files.wrappers import standStraightReward
from gym_files.wrappers import observationFilter
from gym_files.wrappers import actionValueScaler
from tf_agents_local.models.ddpg import DDPG

base_env = gym.make("gym_files/BasicSkateboardEnv-v0", mode = "direct", gravity=10)
gym_env = observationFilter(standStraightReward(actionValueScaler(base_env)))
py_env = gym_wrapper.GymWrapper(gym_env)
env = tf_py_environment.TFPyEnvironment(py_env)

actor_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-3,
    decay_steps=100000,
    decay_rate=0.98)
critic_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-5,
    decay_steps=100000,
    decay_rate=0.98)
ACTOR_FC_LAYERS = (64, 64, 32, 16)
CRITIC_FC_LAYERS = (256, 256, 128, 128, 64, 64, 32, 32)

ddpg_model = DDPG(env, actor_fc_layers=ACTOR_FC_LAYERS,
                  critic_fc_layers=CRITIC_FC_LAYERS,
                  checkpoint_dir="tf_agents_files/checkpoint",
                  log_dir="tf_agents_files/log_files/",
                  actor_learning_rate=actor_lr,
                  critic_learning_rate=critic_lr,
                  ou_stddev=0.8, ou_damping=0.3, target_update_tau=0.03, target_update_period=7,
                  dqda_clipping=None, tf_errors_loss_fn=tf.compat.v1.losses.huber_loss, gamma=0.98,
                  reward_scale_factor=1, gradient_clipping=0.2, batch_size=128,
                  max_buffer_length=10000,initial_collection_steps=1000,
                  reset_collection_steps=200, collection_per_interation=1000)


ddpg_model.train()
