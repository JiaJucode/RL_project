from typing import Optional, Text
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.utils import common
from tf_agents.typing import types
from tf_agents.agents import data_converter
from tf_agents.utils import nest_utils
from tf_agents.utils import eager_utils
from math import log
import numpy as np

class CaclaAgent(tf_agent.TFAgent):
    """A CACLA Agent."""
    
    def __init__(self,
                time_step_spec: ts.TimeStep,
                action_spec: types.NestedTensorSpec,
                actor_network: network.Network,
                critic_network: network.Network,
                actor_optimizer: Optional[types.Optimizer] = None,
                critic_optimizer: Optional[types.Optimizer] = None,
                ou_stddev: types.Float = 1.0,
                ou_damping: types.Float = 1.0,
                target_actor_network: Optional[network.Network] = None,
                target_critic_network: Optional[network.Network] = None,
                target_update_tau: types.Float = 1.0,
                target_update_period: types.Int = 1,
                dqda_clipping: Optional[types.Float] = None,
                td_errors_loss_fn: Optional[types.LossFn] = None,
                gamma: types.Float = 1.0,
                reward_scale_factor: types.Float = 1.0,
                gradient_clipping: Optional[types.Float] = None,
                debug_summaries: bool = False,
                summarize_grads_and_vars: bool = False,
                train_step_counter: Optional[tf.Variable] = None,
                td_error_update_rate: types.Float = 0.01,
                name: Optional[Text] = None):
        tf.Module.__init__(self, name=name)
        self._actor_network = actor_network
        actor_network.create_variables(
            time_step_spec.observation)
        if target_actor_network:
            target_actor_network.create_variables(time_step_spec.observation)
        self._target_actor_network = common.maybe_copy_target_network_with_checks(
            self._actor_network,
            target_actor_network,
            'TargetActorNetwork',
            input_spec=time_step_spec.observation)

        self._critic_network = critic_network
        critic_input_spec = (time_step_spec.observation, action_spec)
        critic_network.create_variables(critic_input_spec)
        if target_critic_network:
            target_critic_network.create_variables(critic_input_spec)
        self._target_critic_network = common.maybe_copy_target_network_with_checks(
            self._critic_network,
            target_critic_network,
            'TargetCriticNetwork',
            input_spec=critic_input_spec)

        self._actor_optimizer = actor_optimizer
        self._critic_optimizer = critic_optimizer

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._dqda_clipping = dqda_clipping
        self._td_errors_loss_fn = (
            td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping

        def update():
            critic_update = common.soft_variables_update(
                self._critic_network.variables,
                self._target_critic_network.variables,
                target_update_tau,
                tau_non_trainable=1.0)
            actor_update = common.soft_variables_update(
                self._actor_network.variables,
                self._target_actor_network.variables,
                target_update_tau,
                tau_non_trainable=1.0)
            return tf.group(critic_update, actor_update)

        self._update_target = common.Periodically(
            update, target_update_period, 'periodic_update_targets')

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)
        collect_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)
        collect_policy = ou_noise_policy.OUNoisePolicy(
            collect_policy,
            ou_stddev=self._ou_stddev,
            ou_damping=self._ou_damping,
            clip=True)

        super(CaclaAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=2 if not self._actor_network.state_spec else None,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=not self._actor_network.state_spec)
        # TODO: inplement variance for multiple updates(td / root td_var) 
        self.td_error_var = 1.0
        self.td_error_update_rate = td_error_update_rate

    def _initialize(self):
        common.soft_variables_update(
            self._critic_network.variables,
            self._target_critic_network.variables,
            tau=1.0)
        common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

    def _loss(self, experience, weights=None, training=False):
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        critic_loss, td_errors = self._critic_loss(
            time_steps, actions, next_time_steps, weights=weights)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')

        actor_loss = self._actor_loss(
            td_errors, time_steps, actions, weights=weights)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

        return tf_agent.LossInfo(actor_loss + critic_loss,
                            (actor_loss, critic_loss))

    def _train(self, experience, weights=None):
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        critic_loss, td_errors = self._policy_eval(
            time_steps, actions, next_time_steps, weights=weights)

        if tf.reduce_any(td_errors > 0):
            actor_loss = self._policy_update(
                td_errors, time_steps, actions, weights=weights)
        else :
            actor_loss = tf.constant(0.0)

        self.train_step_counter.assign_add(1)
        self._update_target()

        return tf_agent.LossInfo(actor_loss + critic_loss,
                            (actor_loss, critic_loss))

    def _policy_eval(self, time_steps, actions, next_time_steps, weights=None):
        trainable_critic_variables = self._critic_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, \
                ('No trainable critic variables to optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss, td_error = self._critic_loss(
                time_steps, actions, next_time_steps, weights=weights)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables, self._critic_optimizer)

        return critic_loss, td_error

    def _policy_update(self, td_errors, time_steps, actions, weights=None):
        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, \
                ('No trainable actor variables to optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self._actor_loss(
                td_errors, time_steps, actions, weights=weights)
            if self._debug_summaries:
                common.generate_tensor_summaries('actor_loss', actor_loss,
                                                self.train_step_counter)

        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables, self._actor_optimizer)
        return actor_loss

    def _apply_gradients(self, gradients, variables, optimizer):
        grads_and_vars = tuple(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars,self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(
                grads_and_vars,self.train_step_counter)
            eager_utils.add_gradients_summaries(
                grads_and_vars,self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _actor_loss(self, td_errors, time_steps, actions_took, weights=None):
        target_actions = actions_took[td_errors > 0]
        network_actions, _ = self._actor_network(
            time_steps.observation, training=False)
        actions = network_actions[td_errors > 0]
        loss = common.element_wise_squared_loss(target_actions, actions)
        if nest_utils.is_batched_nested_tensors(
            time_steps, self.time_step_spec, num_outer_dims=2):
            loss = tf.reduce_sum(loss, axis=1)
        if weights is not None:
            loss *= weights
        actor_loss = tf.reduce_mean(loss)
        with tf.name_scope('Losses/'):
            tf.compat.v2.summary.scalar(
                name='actor_loss', data=actor_loss, step=self.train_step_counter)
        return actor_loss

    def _critic_loss(self, time_steps, actions, next_time_steps, weights=None):
        q_values, _ = self._critic_network(
            (time_steps.observation, actions), step_type=time_steps.step_type,
            training=False)
        target_actions, _ = self._target_actor_network(
            next_time_steps.observation, step_type=next_time_steps.step_type,
            training=False)
        target_q_values, _ = self._target_critic_network(
            (next_time_steps.observation, target_actions),
            step_type=next_time_steps.step_type,
            training=False)
        td_targets = tf.stop_gradient(
            self._reward_scale_factor * next_time_steps.reward +
            self._gamma * next_time_steps.discount * target_q_values)
        td_errors = td_targets - q_values
        critic_loss = self._td_errors_loss_fn(td_targets, q_values)
        if nest_utils.is_batched_nested_tensors(
            time_steps, self.time_step_spec, num_outer_dims=2):
            critic_loss = tf.reduce_sum(critic_loss, axis=1)
        if weights is not None:
            critic_loss *= weights
        critic_loss = tf.reduce_mean(critic_loss)
        with tf.name_scope('Losses/'):
            tf.compat.v2.summary.scalar(
                name='critic_loss', data=critic_loss, step=self.train_step_counter)

        if self._debug_summaries:
            common.generate_tensor_summaries('td_errors', td_errors,
                                            self.train_step_counter)
            common.generate_tensor_summaries('td_targets', td_targets,
                                            self.train_step_counter)
            common.generate_tensor_summaries('q_values', q_values,
                                            self.train_step_counter)
        return critic_loss, td_errors
