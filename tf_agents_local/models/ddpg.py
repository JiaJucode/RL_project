import tensorflow as tf
from collections import deque
import numpy as np
from tf_agents.agents import ddpg
from tf_agents.agents import DdpgAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

class DDPG():
    def __init__(self,
                 env,
                 actor_fc_layers,
                 critic_fc_layers,
                 checkpoint_dir,
                 log_dir,
                 actor_learning_rate=
                 tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-4,
                    decay_steps=1000,
                    decay_rate=0.99),
                 critic_learning_rate=
                 tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-4,
                    decay_steps=1000,
                    decay_rate=0.99),
                 ou_stddev=1,
                 ou_damping=1,
                 target_update_tau=0.01,
                 target_update_period=7,
                 dqda_clipping=None,
                 tf_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                 gamma=0.9,
                 reward_scale_factor=0.8,
                 gradient_clipping=0.2,
                 batch_size=128,
                 max_buffer_length=100000,
                 initial_collection_steps=1000,
                 reset_collection_steps=200,
                 collection_per_interation=100,
                 ) -> None:

        self.log_dir = log_dir
        self._env = env
        self._batch_size = batch_size
        
        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_learning_rate)
        critic_optimizer = tf.keras.optimizers.Nadam(learning_rate=critic_learning_rate)
        actor_net = ddpg.actor_network.ActorNetwork(
            observation_spec, action_spec, fc_layer_params=actor_fc_layers,
            activation_fn=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            last_kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.1, maxval=0.1)
        )
        critic_net = ddpg.critic_network.CriticNetwork(
            (observation_spec, action_spec), joint_fc_layer_params=critic_fc_layers,
            activation_fn=tf.keras.activations.linear,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            last_kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.01, maxval=0.01)
        )
        train_step_counter = tf.compat.v1.train.get_or_create_global_step()
        self._agent = DdpgAgent(
            env.time_step_spec(),
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            ou_stddev=ou_stddev,
            ou_damping=ou_damping,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            dqda_clipping=dqda_clipping,
            td_errors_loss_fn=tf_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            train_step_counter=train_step_counter,
        )
        self._agent.initialize()

        replay_spec = self._agent.collect_data_spec
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            replay_spec, batch_size=env.batch_size, max_length=max_buffer_length)

        self._agent.train_step_counter.assign(0)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=4, sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self._iterator = iter(dataset)
        
        self._initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            env, self._agent.collect_policy, observers=[replay_buffer.add_batch],
            num_steps=initial_collection_steps)

        self._reset_collect_driver = dynamic_step_driver.DynamicStepDriver(
            env, self._agent.collect_policy, observers=[replay_buffer.add_batch],
            num_steps=reset_collection_steps)

        self._collection_per_interation = collection_per_interation
        self._collect_driver = dynamic_step_driver.DynamicStepDriver(
            env, self._agent.collect_policy, observers=[replay_buffer.add_batch],
            num_steps=collection_per_interation)
        
        self._checkpointer = common.Checkpointer(
            ckpt_dir = checkpoint_dir,
            max_to_keep = 5000,
            agent = self._agent,
            global_step = self._agent.train_step_counter
        )
        self._checkpointer.initialize_or_restore()
        
    def train(self,
              train_iters=120,
              train_per_loop=1000,
              log_interval=1,
              eval_interval=1,
              reset_interval=1,
              reset_increment=0.25):
        self._initial_collect_driver.run()
        self._env.reset()
        print("start")
        next_reset = reset_interval
        returns = []
        critic_losses = []
        actor_losses = []
        past_returns = deque(maxlen=train_per_loop)
        for _ in range(train_iters):
            self._collect_driver.run()
            critic_loss = 0
            actor_loss = 0
            for _ in range(train_per_loop):
                experience, _ = next(self._iterator)
                loss_info = self._agent.train(experience)
                critic_loss += loss_info.extra.critic_loss
                actor_loss += loss_info.extra.actor_loss
                past_returns.append(
                    np.sum(experience.reward.numpy())/self._batch_size/2)
            critic_loss /= train_per_loop
            actor_loss /= train_per_loop
            step = self._agent.train_step_counter.numpy()
            if step % (log_interval * self._collection_per_interation) == 0:
                print(f'step = {step}: critic loss = {critic_loss}')
                print(f'step = {step}: actor loss = {actor_loss}')
                ave = sum(past_returns) / len(past_returns)
                big = max(past_returns)
                small = min(past_returns)
                print(f'step = {step}: Average Return = {ave}, max = {big}, min = {small}')
                returns.append(ave)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                with open(self.log_dir + "returns.txt", "a") as f:
                    f.write(str(ave) + "\n")
                    f.flush()
                with open(self.log_dir + "critic_loss.txt", "a") as f:
                    f.write(str(critic_loss.numpy()) + "\n")
                    f.flush()
                with open(self.log_dir + "actor_loss.txt", "a") as f:
                    f.write(str(actor_loss.numpy()) + "\n")
                    f.flush()
            if step % (eval_interval * train_per_loop) == 0:
                self._checkpointer.save(global_step=step)
            # if step % next_reset == 0:
            #     reset_interval += reset_increment
            #     next_reset += reset_interval
            #     self._env.reset()
            #     self._reset_collect_driver.run()
            print(step)
            print(next_reset * train_per_loop)
            if step >= (next_reset * train_per_loop):
                self._env.reset()
                reset_interval += reset_increment
                next_reset += reset_interval
        return returns, critic_losses, actor_losses
                