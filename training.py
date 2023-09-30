# import numpy as np
# import gymnasium as gym
import os
from collections import deque
import gym
import tensorflow as tf
import numpy as np
from tf_agents.utils import common
from tf_agents.agents import ddpg
from tf_agents.agents import DdpgAgent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.eval import metric_utils

from gym_files.wrappers import standStraightReward
from gym_files.wrappers import observationFilter
from gym_files.wrappers import stayBalanceReward
from tf_agents_local.agents.cacla_agent import CaclaAgent


base_env = gym.make("gym_files/BasicSkateboardEnv-v0", mode = "direct", gravity=10)
# gym_env = observationFilter(standStraightReward(base_env))
gym_env = observationFilter(stayBalanceReward(base_env))
py_env = gym_wrapper.GymWrapper(gym_env)
env = tf_py_environment.TFPyEnvironment(py_env)

# agent
INITIAL_LEARNING_RATE = 1e-3
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=3000,
    decay_rate=0.99)
# ACTOR_FC_LAYERS = (512, 512, 256, 64)
# CRITIC_FC_LAYERS = (512, 512, 256, 64)
ACTOR_FC_LAYERS = (64,)
CRITIC_FC_LAYERS = (64,)
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
td_errors_loss_fn=tf.keras.losses.MeanSquaredError()
gamma=0.95
reward_scale_factor=1
gradient_clipping=0.2
TRAIN_STEP_COUNTER = tf.compat.v1.train.get_or_create_global_step()
agent = CaclaAgent(
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
BATCH_SIZE = 128
MAX_LENGTH = 10
replay_spec = agent.collect_data_spec
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    replay_spec, batch_size=env.batch_size, max_length=MAX_LENGTH)

agent.train_step_counter.assign(0)

# data collection
train_per_loop = 1
dataset = replay_buffer.as_dataset(
    num_parallel_calls=4, sample_batch_size=BATCH_SIZE,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

returns = []
train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
        ]
# eval_metrics = [
#     tf_metrics.AverageReturnMetric(buffer_size=NUM_EVAL_EPISODES),
#     tf_metrics.AverageEpisodeLengthMetric(buffer_size=NUM_EVAL_EPISODES),
# ]
# summaries_flush_secs=10
agent_dir = './tf_agents_files/agents'
# eval_dir = os.path.join(agent_dir, 'eval')
# eval_summary_writer = tf.compat.v2.summary.create_file_writer(
#     eval_dir, flush_millis=summaries_flush_secs * 1000
# )

# driver
initial_collection_steps = BATCH_SIZE
collection_per_interation = 1
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    env, agent.collect_policy, observers=[replay_buffer.add_batch],
    num_steps=initial_collection_steps)

reset_collect_driver = dynamic_step_driver.DynamicStepDriver(
    env, agent.collect_policy, observers=[replay_buffer.add_batch],
    num_steps=200)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    env, agent.collect_policy, observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=collection_per_interation)

# test_driver = dynamic_step_driver.DynamicStepDriver(
#     env, agent.policy,
#     num_steps=BATCH_SIZE)

time_step = None

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

checkpointer.initialize_or_restore()


# training
# init_reset_interval = 1000
# next_reset = init_reset_interval
initial_collect_driver.run()
time_step = None
print("start")
PASR_RETURN_LENGTH = 100
past_returns = deque(maxlen=PASR_RETURN_LENGTH)
with open(agent_dir + "/returns.txt", "a") as f:
    for _ in range(400000):
        time_step, _ = collect_driver.run()
        for _ in range(train_per_loop):
            experience, _ = next(iterator)
            loss_info = agent.train(experience)
        step = agent.train_step_counter.numpy()
        reward = np.sum(experience.reward.numpy())/BATCH_SIZE/2
        past_returns.append(reward)
        if step % LOG_INTERVAL == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss))
            ave = sum(past_returns) / len(past_returns)
            print('step = {0}: Average Return = {1}'
                    .format(step, ave))
            f.write(str(ave) + "\n")
            f.flush()
        if step % EVAL_INTERVAL == 0:
            checkpointer.save(global_step=step)
        # if step % next_reset == 0:
        #     init_reset_interval += 200
        #     next_reset += init_reset_interval
        #     env.reset()
        #     reset_collect_driver.run()


env.close()
