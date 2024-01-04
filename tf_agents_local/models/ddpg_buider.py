from .ddpg import DDPG

class DDPG_builder:
    def __init__(self, env, checkpoint_dir, log_dir) -> None:
        self._env = env
        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir
        self._actor_learning_rate = 1e-4
        self._critic_learning_rate = 1e-4
        self._ou_stddev = 1
        self._ou_damping = 1
        self._target_update_tau = 0.01
        self._target_update_period = 7
        self._dqda_clipping = None
        self._tf_errors_loss_fn = None
        self._gamma = 0.9
        self._reward_scale_factor = 0.8
        self._gradient_clipping = 0.2
        self._batch_size = 128
        self._max_buffer_length = 100
        self._initial_collection_steps = 1000
        self._reset_collection_steps = 200
        self._collection_per_interation = 100
        self._actor_fc_layers = (64, 64, 32, 16)
        self._critic_fc_layers = (64, 64, 32, 16)

    def set_actor_learning_rate(self, actor_learning_rate):
        self._actor_learning_rate = actor_learning_rate
        return self

    def set_critic_learning_rate(self, critic_learning_rate):
        self._critic_learning_rate = critic_learning_rate
        return self

    def set_ou_stddev(self, ou_stddev):
        self._ou_stddev = ou_stddev
        return self

    def set_ou_damping(self, ou_damping):
        self._ou_damping = ou_damping
        return self

    def set_target_update_tau(self, target_update_tau):
        self._target_update_tau = target_update_tau
        return self

    def set_target_update_period(self, target_update_period):
        self._target_update_period = target_update_period
        return self

    def set_gamma(self, gamma):
        self._gamma = gamma
        return self

    def set_checkpoint_dir(self, checkpoint_dir):
        self._checkpoint_dir = checkpoint_dir
        return self

    def set_log_dir(self, log_dir):
        self._log_dir = log_dir
        return self

    def build(self):
        return DDPG(self._env, self._actor_fc_layers, self._critic_fc_layers,
                    self._checkpoint_dir, self._log_dir,
                    self._actor_learning_rate, self._critic_learning_rate,
                    self._ou_stddev, self._ou_damping,
                    self._target_update_tau, self._target_update_period,
                    self._dqda_clipping, self._tf_errors_loss_fn,
                    self._gamma, self._reward_scale_factor,
                    self._gradient_clipping, self._batch_size,
                    self._max_buffer_length, self._initial_collection_steps,
                    self._reset_collection_steps, self._collection_per_interation)
