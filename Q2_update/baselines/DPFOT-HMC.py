"""DPFOT-HMC: OT resampling followed by Hamiltonian Monte Carlo transitions."""

from __future__ import annotations

import tensorflow as tf

from dataset import observation_mean_tf, transition_drift_tf


class DPFOT_HMC:
	"""DPFOT with post-transport Hamiltonian Monte Carlo exploration."""

	def __init__(
		self,
		state_dim: int,
		num_particles: int,
		process_var: float,
		obs_var: float,
		init_var: float,
		epsilon: float = 0.1,
		sinkhorn_iters: int = 50,
		hmc_steps: int = 3,
		hmc_leapfrog_steps: int = 3,
		hmc_step_size: float = 0.02,
	) -> None:
		self.state_dim = int(state_dim)
		self.num_particles = int(num_particles)
		self.process_var = tf.constant(process_var, dtype=tf.float32)
		self.obs_var = tf.constant(obs_var, dtype=tf.float32)
		self.init_var = tf.constant(init_var, dtype=tf.float32)
		self.epsilon = float(epsilon)
		self.sinkhorn_iters = int(sinkhorn_iters)
		self.hmc_steps = int(hmc_steps)
		self.hmc_leapfrog_steps = int(hmc_leapfrog_steps)
		self.hmc_step_size = float(hmc_step_size)

	@staticmethod
	def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
		return logw - tf.reduce_logsumexp(logw)

	@staticmethod
	def _ess(logw: tf.Tensor) -> tf.Tensor:
		return tf.exp(-tf.reduce_logsumexp(2.0 * logw))

	def _transition_sample(self, particles: tf.Tensor, time_step: tf.Tensor, seed: int) -> tf.Tensor:
		drift = transition_drift_tf(particles, time_step)
		noise = tf.random.stateless_normal(tf.shape(particles), seed=(seed, 61), stddev=tf.sqrt(self.process_var), dtype=particles.dtype)
		return drift + noise

	def _observation_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		loc = observation_mean_tf(particles)
		log_prob_dim = -0.5 * (
			tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=particles.dtype) * self.obs_var)
			+ tf.square(observation[tf.newaxis, :] - loc) / self.obs_var
		)
		return tf.reduce_sum(log_prob_dim, axis=-1)

	def _sinkhorn_transport(self, particles: tf.Tensor, logw: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		n = self.num_particles
		n_f = tf.cast(n, particles.dtype)
		log_a = tf.fill([n], -tf.math.log(n_f))
		log_b = self._normalize_logw(logw)

		x_i = particles[:, tf.newaxis, :]
		x_j = particles[tf.newaxis, :, :]
		cost = tf.reduce_sum(tf.square(x_i - x_j), axis=-1)

		eps = tf.cast(self.epsilon, particles.dtype)
		f = tf.zeros_like(log_a)
		g = tf.zeros_like(log_b)
		for _ in range(self.sinkhorn_iters):
			f = eps * (log_a - tf.reduce_logsumexp((g[tf.newaxis, :] - cost) / eps, axis=1))
			g = eps * (log_b - tf.reduce_logsumexp((f[:, tf.newaxis] - cost) / eps, axis=0))

		log_p = (f[:, tf.newaxis] + g[tf.newaxis, :] - cost) / eps
		log_p = tf.clip_by_value(log_p, -40.0, 10.0)
		p = tf.exp(log_p)
		x_new = tf.matmul(p * n_f, particles)
		x_new = tf.where(tf.math.is_finite(x_new), x_new, particles)
		return x_new, log_a

	def _grad_log_prob(self, particles: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
		with tf.GradientTape() as tape:
			tape.watch(particles)
			logp = self._observation_log_prob(particles, observation)
			target = tf.reduce_sum(logp)
		grad = tape.gradient(target, particles)
		if grad is None:
			grad = tf.zeros_like(particles)
		return tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(particles))

	def _hmc_transition(self, particles: tf.Tensor, observation: tf.Tensor, seed: int) -> tf.Tensor:
		eps = tf.cast(self.hmc_step_size, particles.dtype)
		x = particles
		for s in range(self.hmc_steps):
			p0 = tf.random.stateless_normal(tf.shape(x), seed=(seed, 700 + s), dtype=x.dtype)
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
			u = tf.random.stateless_uniform(tf.shape(accept_prob), seed=(seed, 900 + s), dtype=x.dtype)
			accept = u < accept_prob
			x = tf.where(accept[:, tf.newaxis], x_prop, x)

		return x

	def initialize(self, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
		particles = tf.random.stateless_normal((self.num_particles, self.state_dim), seed=(seed, 6), stddev=tf.sqrt(self.init_var), dtype=tf.float32)
		logw = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
		return particles, logw

	def step(self, particles: tf.Tensor, log_weights: tf.Tensor, observation: tf.Tensor, time_step: tf.Tensor, seed: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
		particles = self._transition_sample(particles, time_step, seed=seed)
		logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
		ess = self._ess(logw)
		transported, lw = self._sinkhorn_transport(particles, logw)
		hmc_particles = self._hmc_transition(transported, observation, seed=seed)
		return hmc_particles, lw, ess, tf.constant(1.0, dtype=tf.float32)
