# import numpy as np
# import gymnasium as gym
import os
import gym
import sys
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.agents import ddpg
from tf_agents.agents import DdpgAgent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval import metric_utils

from gym_files.wrappers import standStraightReward
from gym_files.wrappers import observationFilter
from tf_agents_local.agents.cacla_agent import CaclaAgent

args = sys.argv
test_step = -1
if len(args) > 1:
    test_step = int(args[1])

INITIAL_COLLECT_STEPS = 1000 # @param {type:"integer"}
COLLECT_STEPS_PER_ITERATION = 1 # @param {type:"integer"}
REPLAY_BUFFER_MAX_LENGTH = 100000 # @param {type:"integer"}

BATCH_SIZE = 64 # @param {type:"integer"}
LOG_INTERVAL = 200 # @param {type:"integer"}

NUM_EVAL_EPISODES = 10 # @param {type:"integer"}
EVAL_INTERVAL = 1000 # @param {type:"integer"}


base_env = gym.make("gym_files/BasicSkateboardEnv-v0", mode = "gui", gravity=10)
gym_env = observationFilter(standStraightReward(base_env))
py_env = gym_wrapper.GymWrapper(gym_env)
env = tf_py_environment.TFPyEnvironment(py_env)

# agent
INITIAL_LEARNING_RATE = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.9)
# ACTOR_FC_LAYERS = (512, 512, 256, 64)
# CRITIC_FC_LAYERS = (512, 512, 256, 64)
ACTOR_FC_LAYERS = (64, 64, 32, 16)
CRITIC_FC_LAYERS = (256, 256, 128, 128, 64, 64, 32, 32)
ACTOR_OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
CRITIC_OPTIMIZER = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)


action_spec = env.action_spec()
observation_spec = env.observation_spec()

actor_net = ddpg.actor_network.ActorNetwork(
    observation_spec, action_spec, fc_layer_params=ACTOR_FC_LAYERS,
    activation_fn=tf.keras.activations.tanh,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003),
    last_kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.01, maxval=0.01)
)
critic_net = ddpg.critic_network.CriticNetwork(
    (observation_spec, action_spec), joint_fc_layer_params=CRITIC_FC_LAYERS,
    activation_fn=tf.keras.activations.tanh,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003),
    last_kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.01, maxval=0.01)
)

ou_stddev=1.2
ou_damping=0.3
target_update_tau=0.03
target_update_period=7
dqda_clipping=None
td_errors_loss_fn=tf.keras.losses.KLDivergence()
gamma=0.95
reward_scale_factor=1
gradient_clipping=0.2
TRAIN_STEP_COUNTER = tf.compat.v1.train.get_or_create_global_step()
agent = DdpgAgent(
    env.time_step_spec(),
    env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=ACTOR_OPTIMIZER,
    critic_optimizer=CRITIC_OPTIMIZER,
    ou_stddev=ou_stddev,
    # ou_damping=ou_damping,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    dqda_clipping=dqda_clipping,
    td_errors_loss_fn=td_errors_loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=TRAIN_STEP_COUNTER,
)
agent.initialize()



# replay buffer
BATCH_SIZE = 32
MAX_LENGTH = 100000
replay_spec = agent.collect_data_spec
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    replay_spec, batch_size=env.batch_size, max_length=MAX_LENGTH)

agent.train_step_counter.assign(0)

returns = []
train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
        ]
agent_dir = './tf_agents_files'

# agent saving
LOG_INTERVAL = 200
EVAL_INTERVAL = 1000
checkpoint_dir = os.path.join(agent_dir, 'checkpoint')
checkpointer = common.Checkpointer(
    ckpt_dir = checkpoint_dir,
    max_to_keep = 5000,
    agent = agent,
    global_step = agent.train_step_counter,
    metric = metric_utils.MetricsGroup(train_metrics, 'train_metrics')
)

if test_step != -1:
    checkpointer.initialize_or_restore(test_step)
else:
    checkpointer.initialize_or_restore()
print("start")
time_step = env.reset()
while True:
    action_step = agent.collect_policy.action(time_step)
    # print(action_step.action)
    time_step = env.step(action_step.action)
    print(time_step.reward)
    print()


env.close()
