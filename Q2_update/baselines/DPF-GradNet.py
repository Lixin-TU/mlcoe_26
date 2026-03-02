"""DPF-GradNet: amortized transport resampling using a conditional neural map."""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class DPF_GradNet:
	"""Amortized OT-style resampling baseline using a single GradNet."""

	def __init__(
		self,
		state_dim: int,
		num_particles: int,
		process_var: float,
		obs_var: float,
		init_var: float,
		hidden_units: int = 64,
		transport_strength: float = 0.35,
	) -> None:
		self.state_dim = int(state_dim)
		self.num_particles = int(num_particles)
		self.process_var = tf.constant(process_var, dtype=tf.float32)
		self.obs_var = tf.constant(obs_var, dtype=tf.float32)
		self.init_var = tf.constant(init_var, dtype=tf.float32)
		self.transport_strength = float(transport_strength)
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
		noise = tf.random.stateless_normal(tf.shape(particles), seed=(seed, 71), stddev=tf.sqrt(self.process_var), dtype=particles.dtype)
		return drift + noise

	def _observation_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		loc = observation_mean_tf(particles)
		log_prob_dim = -0.5 * (
			tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=particles.dtype) * self.obs_var)
			+ tf.square(observation[tf.newaxis, :] - loc) / self.obs_var
		)
		return tf.reduce_sum(log_prob_dim, axis=-1)

	def initialize(self, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
		particles = tf.random.stateless_normal((self.num_particles, self.state_dim), seed=(seed, 7), stddev=tf.sqrt(self.init_var), dtype=tf.float32)
		logw = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return particles, logw

	def step(self, particles: tf.Tensor, log_weights: tf.Tensor, observation: tf.Tensor, time_step: tf.Tensor, seed: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		particles = self._transition_sample(particles, time_step, seed=seed)
		logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
		ess = self._ess(logw)
		w = tf.exp(logw)
		context = tf.concat([particles, w[:, tf.newaxis]], axis=-1)
		net_disp = 0.05 * tf.tanh(self.grad_net(context))
		weighted_mean = tf.reduce_sum(particles * w[:, tf.newaxis], axis=0, keepdims=True)
		pull_disp = self.transport_strength * (weighted_mean - particles)
		x_new = particles + pull_disp + net_disp
		lw_new = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return x_new, lw_new, ess, tf.constant(1.0, dtype=tf.float32)
