"""DPFS: Differentiable Particle Filter with soft-resampling."""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class DPFS:
	"""Soft-resampling particle filter baseline.

	Uses mixture resampling between weighted and uniform distributions with
	importance correction to preserve unbiased estimates.
	"""

	def __init__(self, state_dim: int, num_particles: int, process_var: float, obs_var: float, init_var: float, alpha: float = 0.7) -> None:
		self.state_dim = int(state_dim)
		self.num_particles = int(num_particles)
		self.process_var = tf.constant(process_var, dtype=tf.float32)
		self.obs_var = tf.constant(obs_var, dtype=tf.float32)
		self.init_var = tf.constant(init_var, dtype=tf.float32)
		self.alpha = float(alpha)

	@staticmethod
	def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
		return logw - tf.reduce_logsumexp(logw)

	@staticmethod
	def _ess(logw: tf.Tensor) -> tf.Tensor:
		return tf.exp(-tf.reduce_logsumexp(2.0 * logw))

	def _transition_sample(self, particles: tf.Tensor, time_step: tf.Tensor, seed: int) -> tf.Tensor:
		drift = transition_drift_tf(particles, time_step)
		noise = tf.random.stateless_normal(tf.shape(particles), seed=(seed, 21), stddev=tf.sqrt(self.process_var), dtype=particles.dtype)
		return drift + noise

	def _observation_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		loc = observation_mean_tf(particles)
		log_prob_dim = -0.5 * (
			tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=particles.dtype) * self.obs_var)
			+ tf.square(observation[tf.newaxis, :] - loc) / self.obs_var
		)
		return tf.reduce_sum(log_prob_dim, axis=-1)

	def initialize(self, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
		particles = tf.random.stateless_normal((self.num_particles, self.state_dim), seed=(seed, 2), stddev=tf.sqrt(self.init_var), dtype=tf.float32)
		logw = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return particles, logw

	def step(self, particles: tf.Tensor, log_weights: tf.Tensor, observation: tf.Tensor, time_step: tf.Tensor, seed: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		particles = self._transition_sample(particles, time_step, seed=seed)
		logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
		ess = self._ess(logw)

		n_f = tf.cast(self.num_particles, tf.float32)
		w = tf.exp(logw)
		uniform = tf.ones_like(w) / n_f
		mixed = self.alpha * w + (1.0 - self.alpha) * uniform

		indices = tf.random.stateless_categorical(
			logits=tf.math.log(mixed[tf.newaxis, :] + 1e-12),
			num_samples=self.num_particles,
			seed=(seed, 23),
			dtype=tf.int32,
		)[0]
		x_new = tf.gather(particles, indices)
		w_old = tf.gather(w, indices)
		w_mix = tf.gather(mixed, indices)
		logw_new = self._normalize_logw(tf.math.log(w_old / (w_mix * n_f) + 1e-12))
		return x_new, logw_new, ess, tf.constant(1.0, dtype=tf.float32)
