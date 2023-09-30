import gym
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from gym_files.wrappers import standStraightReward
from gym_files.wrappers import observationFilter
from tf_agents_local.models.ddpg import DDPG

base_env = gym.make("gym_files/BasicSkateboardEnv-v0", mode = "direct", gravity=10)
gym_env = observationFilter(standStraightReward(base_env))
py_env = gym_wrapper.GymWrapper(gym_env)
env = tf_py_environment.TFPyEnvironment(py_env)

INITIAL_LEARNING_RATE = 3e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=3000,
    decay_rate=0.99)
ACTOR_FC_LAYERS = (512, 512, 256, 64, 32)
CRITIC_FC_LAYERS = (512, 512, 256, 64, 32)

ddpg_model = DDPG(env, actor_fc_layers=(512, 512, 256, 64, 32),
                  critic_fc_layers=(512, 512, 256, 64, 32),
                  checkpoint_dir="tf_agents_files/checkpoint",
                  reward_log_dir="tf_agents_files/log_files/test1",
                  actor_learning_rate=lr_schedule,
                  critic_learning_rate=lr_schedule,
                  ou_stddev=1, ou_damping=0.3, target_update_tau=0.03, target_update_period=7,
                  dqda_clipping=None, tf_errors_loss_fn=tf.compat.v1.losses.huber_loss, gamma=0.95,
                  reward_scale_factor=1, gradient_clipping=0.2, batch_size=128,
                  max_buffer_length=100000,initial_collection_steps=1000,
                  reset_collection_steps=200, collection_per_interation=1)

ddpg_model.start()
