"""IPFPF: Invertible Particle Flow Particle Filter (deterministic LEDH-style flow)."""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class IPFPF:
	"""Deterministic particle-flow baseline with uniform post-flow weights."""

	def __init__(
		self,
		state_dim: int,
		num_particles: int,
		process_var: float,
		obs_var: float,
		init_var: float,
		flow_steps: int = 10,
		flow_step_size: float = 0.08,
	) -> None:
		self.state_dim = int(state_dim)
		self.num_particles = int(num_particles)
		self.process_var = tf.constant(process_var, dtype=tf.float32)
		self.obs_var = tf.constant(obs_var, dtype=tf.float32)
		self.init_var = tf.constant(init_var, dtype=tf.float32)
		self.flow_steps = int(flow_steps)
		self.flow_step_size = float(flow_step_size)

	@staticmethod
	def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
		return logw - tf.reduce_logsumexp(logw)

	@staticmethod
	def _ess(logw: tf.Tensor) -> tf.Tensor:
		return tf.exp(-tf.reduce_logsumexp(2.0 * logw))

	def _transition_sample(self, particles: tf.Tensor, time_step: tf.Tensor, seed: int) -> tf.Tensor:
		drift = transition_drift_tf(particles, time_step)
		noise = tf.random.stateless_normal(tf.shape(particles), seed=(seed, 41), stddev=tf.sqrt(self.process_var), dtype=particles.dtype)
		return drift + noise

	def _observation_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		loc = observation_mean_tf(particles)
		log_prob_dim = -0.5 * (
			tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=particles.dtype) * self.obs_var)
			+ tf.square(observation[tf.newaxis, :] - loc) / self.obs_var
		)
		return tf.reduce_sum(log_prob_dim, axis=-1)

	def _flow_update(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		x = particles
		lam = tf.cast(self.flow_step_size, x.dtype)
		for _ in range(self.flow_steps):
			with tf.GradientTape() as tape:
				tape.watch(x)
				ll = self._observation_log_prob(x, observation)
			grad = tape.gradient(tf.reduce_sum(ll), x)
			if grad is None:
				grad = tf.zeros_like(x)
			grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(x))
			x = x + 0.5 * lam * grad
		return x

	def initialize(self, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
		particles = tf.random.stateless_normal((self.num_particles, self.state_dim), seed=(seed, 4), stddev=tf.sqrt(self.init_var), dtype=tf.float32)
		logw = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return particles, logw

	def step(self, particles: tf.Tensor, log_weights: tf.Tensor, observation: tf.Tensor, time_step: tf.Tensor, seed: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		particles = self._transition_sample(particles, time_step, seed=seed)
		logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
		ess = self._ess(logw)
		x_new = self._flow_update(particles, observation)
		lw_new = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return x_new, lw_new, ess, tf.constant(1.0, dtype=tf.float32)
