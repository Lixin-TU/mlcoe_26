"""DPFNet-PMMH: GradNet transport followed by PMMH-style transitions.

This model combines amortized transport (DPF-GradNet style) with post-transport
Pseudo-Marginal Metropolis-Hastings transitions to improve particle diversity.
"""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class DPFNet_PMMH:
    """DPF-GradNet + PMMH baseline.

    Pipeline per step:
    1) Predict with transition model.
    2) Importance weight update from observation likelihood.
    3) GradNet residual transport (amortized map).
    4) PMMH-style random-walk transitions with pseudo-marginal acceptance.
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
        pmmh_steps: int = 3,
        pmmh_proposal_std: float = 0.02,
        pmmh_inner_samples: int = 4,
        pmmh_likelihood_jitter: float = 0.10,
        net_scale: float = 0.04,
    ) -> None:
        self.state_dim = int(state_dim)
        self.num_particles = int(num_particles)
        self.process_var = tf.constant(process_var, dtype=tf.float32)
        self.obs_var = tf.constant(obs_var, dtype=tf.float32)
        self.init_var = tf.constant(init_var, dtype=tf.float32)
        self.transport_strength = float(transport_strength)
        self.pmmh_steps = int(pmmh_steps)
        self.pmmh_proposal_std = float(pmmh_proposal_std)
        self.pmmh_inner_samples = max(1, int(pmmh_inner_samples))
        self.pmmh_likelihood_jitter = float(pmmh_likelihood_jitter)
        self.net_scale = float(net_scale)
        self.last_accept_rate = tf.constant(0.0, dtype=tf.float32)
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

    def _pseudo_marginal_loglikelihood(self, particles: tf.Tensor, observation: tf.Tensor, seed: int) -> tf.Tensor:
        if self.pmmh_inner_samples <= 1:
            return self._observation_log_prob(particles, observation)

        d = tf.shape(particles)[1]
        n = tf.shape(particles)[0]
        proposal_scale = tf.cast(self.pmmh_likelihood_jitter, particles.dtype) * tf.sqrt(self.obs_var)
        noise = tf.random.stateless_normal(
            (self.pmmh_inner_samples, n, d),
            seed=(seed, 2101),
            stddev=proposal_scale,
            dtype=particles.dtype,
        )
        cloud = particles[tf.newaxis, :, :] + noise
        flat_cloud = tf.reshape(cloud, (-1, d))
        logp = self._observation_log_prob(flat_cloud, observation)
        logp = tf.reshape(logp, (self.pmmh_inner_samples, n))
        return tf.reduce_logsumexp(logp, axis=0) - tf.math.log(tf.cast(self.pmmh_inner_samples, particles.dtype))

    def _pmmh_transition(self, particles: tf.Tensor, observation: tf.Tensor, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
        x = particles
        proposal_std = tf.cast(self.pmmh_proposal_std, particles.dtype)
        current_loglik = self._pseudo_marginal_loglikelihood(x, observation, seed=seed + 300)
        accepted_rates: list[tf.Tensor] = []

        for s in range(self.pmmh_steps):
            proposal_noise = tf.random.stateless_normal(
                tf.shape(x),
                seed=(seed, 2400 + s),
                stddev=proposal_std,
                dtype=x.dtype,
            )
            x_prop = x + proposal_noise
            prop_loglik = self._pseudo_marginal_loglikelihood(x_prop, observation, seed=seed + 700 + 17 * s)

            log_alpha = tf.clip_by_value(prop_loglik - current_loglik, -30.0, 0.0)
            u = tf.random.stateless_uniform(tf.shape(log_alpha), seed=(seed, 2600 + s), dtype=x.dtype)
            accept = tf.math.log(tf.maximum(u, tf.constant(1e-8, dtype=x.dtype))) < log_alpha

            x = tf.where(accept[:, tf.newaxis], x_prop, x)
            current_loglik = tf.where(accept, prop_loglik, current_loglik)
            accepted_rates.append(tf.reduce_mean(tf.cast(accept, tf.float32)))

        accept_rate = tf.reduce_mean(tf.stack(accepted_rates)) if accepted_rates else tf.constant(0.0, dtype=tf.float32)
        self.last_accept_rate = accept_rate
        return x, accept_rate

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

        pmmh_particles, accept_rate = self._pmmh_transition(transported, observation, seed=seed)
        lw_new = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
        return pmmh_particles, lw_new, ess, tf.cast(accept_rate, tf.float32)


class DPFNet_HMC(DPFNet_PMMH):
    """Backward-compatible alias for legacy code paths."""
