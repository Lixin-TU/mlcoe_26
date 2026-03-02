"""DPFNet-HMC: GradNet transport followed by Hamiltonian Monte Carlo transitions.

This model combines amortized transport (DPF-GradNet style) with post-transport
Hamiltonian Monte Carlo exploration to improve particle diversity.
"""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class DPFNet_HMC:
    """DPF-GradNet + HMC baseline.

    Pipeline per step:
    1) Predict with transition model.
    2) Importance weight update from observation likelihood.
    3) GradNet residual transport (amortized map).
    4) Hamiltonian Monte Carlo transitions on transported particles.
    5) Reset to uniform weights.
    """

    def __init__(
        self,
        state_dim: int,
        num_particles: int,
        process_var: float,
        obs_var: float,
        init_var: float,
        hidden_units: int = 64,
        transport_strength: float = 0.35,
        hmc_steps: int = 3,
        hmc_leapfrog_steps: int = 3,
        hmc_step_size: float = 0.01,
        net_scale: float = 0.04,
    ) -> None:
        self.state_dim = int(state_dim)
        self.num_particles = int(num_particles)
        self.process_var = tf.constant(process_var, dtype=tf.float32)
        self.obs_var = tf.constant(obs_var, dtype=tf.float32)
        self.init_var = tf.constant(init_var, dtype=tf.float32)
        self.transport_strength = float(transport_strength)
        self.hmc_steps = int(hmc_steps)
        self.hmc_leapfrog_steps = int(hmc_leapfrog_steps)
        self.hmc_step_size = float(hmc_step_size)
        self.net_scale = float(net_scale)
        self.grad_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(
                    self.state_dim,
                    kernel_initializer=tf.keras.initializers.Zeros(),
                    bias_initializer=tf.keras.initializers.Zeros(),
                ),
            ]
        )

    @staticmethod
    def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
        return logw - tf.reduce_logsumexp(logw)

    @staticmethod
    def _ess(logw: tf.Tensor) -> tf.Tensor:
        return tf.exp(-tf.reduce_logsumexp(2.0 * logw))

    def _transition_sample(self, particles: tf.Tensor, time_step: tf.Tensor, seed: int) -> tf.Tensor:
        drift = transition_drift_tf(particles, time_step)
        noise = tf.random.stateless_normal(
            tf.shape(particles),
            seed=(seed, 81),
            stddev=tf.sqrt(self.process_var),
            dtype=particles.dtype,
        )
        return drift + noise

    def _observation_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        loc = observation_mean_tf(particles)
        log_prob_dim = -0.5 * (
            tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=particles.dtype) * self.obs_var)
            + tf.square(observation[tf.newaxis, :] - loc) / self.obs_var
        )
        return tf.reduce_sum(log_prob_dim, axis=-1)

    def _grad_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(particles)
            logp = self._observation_log_prob(particles, observation)
            target = tf.reduce_sum(logp)
        grad = tape.gradient(target, particles)
        if grad is None:
            grad = tf.zeros_like(particles)
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)
        return tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(particles))

    def _hmc_transition(self, particles: tf.Tensor, observation: tf.Tensor, seed: int) -> tf.Tensor:
        eps = tf.cast(self.hmc_step_size, particles.dtype)
        x = particles
        for s in range(self.hmc_steps):
            p0 = tf.random.stateless_normal(tf.shape(x), seed=(seed, 1000 + s), dtype=x.dtype)
            x_prop = x
            p = p0

            grad = self._grad_log_prob(x_prop, observation)
            p = p + 0.5 * eps * grad
            for lf in range(self.hmc_leapfrog_steps):
                x_prop = x_prop + eps * p
                grad = self._grad_log_prob(x_prop, observation)
                if lf < self.hmc_leapfrog_steps - 1:
                    p = p + eps * grad
            p = p + 0.5 * eps * grad

            current_logp = self._observation_log_prob(x, observation)
            proposed_logp = self._observation_log_prob(x_prop, observation)
            current_h = -current_logp + 0.5 * tf.reduce_sum(tf.square(p0), axis=-1)
            proposed_h = -proposed_logp + 0.5 * tf.reduce_sum(tf.square(p), axis=-1)
            accept_prob = tf.minimum(1.0, tf.exp(tf.clip_by_value(current_h - proposed_h, -30.0, 0.0)))
            u = tf.random.stateless_uniform(tf.shape(accept_prob), seed=(seed, 1200 + s), dtype=x.dtype)
            accept = u < accept_prob
            x = tf.where(accept[:, tf.newaxis], x_prop, x)

        return x

    def initialize(self, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
        particles = tf.random.stateless_normal(
            (self.num_particles, self.state_dim),
            seed=(seed, 8),
            stddev=tf.sqrt(self.init_var),
            dtype=tf.float32,
        )
        logw = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
        return particles, logw

    def step(
        self,
        particles: tf.Tensor,
        log_weights: tf.Tensor,
        observation: tf.Tensor,
        time_step: tf.Tensor,
        seed: int,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        particles = self._transition_sample(particles, time_step, seed=seed)
        logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
        ess = self._ess(logw)

        weights = tf.exp(logw)
        context = tf.concat([particles, weights[:, tf.newaxis]], axis=-1)
        net_disp = self.net_scale * tf.tanh(self.grad_net(context))
        weighted_mean = tf.reduce_sum(particles * weights[:, tf.newaxis], axis=0, keepdims=True)
        pull_disp = self.transport_strength * (weighted_mean - particles)
        transported = particles + pull_disp + net_disp

        hmc_particles = self._hmc_transition(transported, observation, seed=seed)
        lw_new = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
        return hmc_particles, lw_new, ess, tf.constant(1.0, dtype=tf.float32)
