"""NumPy reference implementation of DPFNet-HMC.

This class mirrors the core API of the TensorFlow variant (`initialize`/`step`),
but uses NumPy and an analytic gradient for the benchmark observation model.
It is intended for reproducible CPU reference runs and benchmarking.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import observation_mean, transition_drift


class DPFNet_HMC_Numpy:
    """DPF-GradNet + HMC reference baseline implemented with NumPy.

    Notes:
    - Uses deterministic mean-pull transport only (no trainable GradNet in NumPy).
    - HMC still targets observation likelihood, matching the TensorFlow logic.
    """

    def __init__(
        self,
        state_dim: int,
        num_particles: int,
        process_var: float,
        obs_var: float,
        init_var: float,
        transport_strength: float = 0.35,
        hmc_steps: int = 3,
        hmc_leapfrog_steps: int = 3,
        hmc_step_size: float = 0.01,
    ) -> None:
        self.state_dim = int(state_dim)
        self.num_particles = int(num_particles)
        self.process_var = float(process_var)
        self.obs_var = float(obs_var)
        self.init_var = float(init_var)
        self.transport_strength = float(transport_strength)
        self.hmc_steps = int(hmc_steps)
        self.hmc_leapfrog_steps = int(hmc_leapfrog_steps)
        self.hmc_step_size = float(hmc_step_size)

    @staticmethod
    def _normalize_logw(logw: np.ndarray) -> np.ndarray:
        m = np.max(logw)
        z = m + np.log(np.sum(np.exp(logw - m)))
        return logw - z

    @staticmethod
    def _ess(logw: np.ndarray) -> float:
        return float(np.exp(-np.log(np.sum(np.exp(2.0 * logw)))))

    def _transition_sample(self, particles: np.ndarray, time_step: int, rng: np.random.Generator) -> np.ndarray:
        drift = transition_drift(particles, int(time_step))
        noise = rng.normal(0.0, np.sqrt(self.process_var), size=particles.shape).astype(np.float32)
        return drift.astype(np.float32) + noise

    def _observation_log_prob(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        loc = observation_mean(particles)
        c = np.log(2.0 * np.pi * self.obs_var)
        return np.sum(-0.5 * (c + ((observation[None, :] - loc) ** 2) / self.obs_var), axis=-1).astype(np.float32)

    def _grad_log_prob(self, particles: np.ndarray, observation: np.ndarray) -> np.ndarray:
        loc = observation_mean(particles)
        resid = observation[None, :] - loc
        dloc_dx = particles / 10.0
        grad = (resid / self.obs_var) * dloc_dx
        return np.where(np.isfinite(grad), grad, 0.0).astype(np.float32)

    def _hmc_transition(self, particles: np.ndarray, observation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eps = float(self.hmc_step_size)
        x = particles.copy()
        for _ in range(self.hmc_steps):
            p0 = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
            x_prop = x.copy()
            p = p0.copy()

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
            current_h = -current_logp + 0.5 * np.sum(p0 * p0, axis=-1)
            proposed_h = -proposed_logp + 0.5 * np.sum(p * p, axis=-1)
            accept_prob = np.minimum(1.0, np.exp(np.clip(current_h - proposed_h, -30.0, 0.0)))
            accept = rng.uniform(0.0, 1.0, size=(x.shape[0],)) < accept_prob
            x[accept] = x_prop[accept]

        return x.astype(np.float32)

    def initialize(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        particles = rng.normal(
            0.0,
            np.sqrt(self.init_var),
            size=(self.num_particles, self.state_dim),
        ).astype(np.float32)
        logw = np.full((self.num_particles,), -np.log(float(self.num_particles)), dtype=np.float32)
        return particles, logw

    def step(
        self,
        particles: np.ndarray,
        log_weights: np.ndarray,
        observation: np.ndarray,
        time_step: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))

        particles = self._transition_sample(particles, time_step=int(time_step), rng=rng)
        logw = self._normalize_logw(log_weights + self._observation_log_prob(particles, observation))
        ess = np.float32(self._ess(logw))

        weights = np.exp(logw)[:, None]
        weighted_mean = np.sum(particles * weights, axis=0, keepdims=True)
        transported = particles + self.transport_strength * (weighted_mean - particles)

        hmc_particles = self._hmc_transition(transported, observation, rng=rng)
        lw_new = np.full((self.num_particles,), -np.log(float(self.num_particles)), dtype=np.float32)
        return hmc_particles, lw_new, ess, np.float32(1.0)
