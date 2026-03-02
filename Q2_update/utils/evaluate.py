"""Evaluation metrics and profiling utilities for particle filtering experiments."""

from __future__ import annotations

import time
from typing import Any, Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math


def _to_tensor(x: Any, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    """Convert arrays or tensors into a TensorFlow tensor with target dtype."""
    return tf.convert_to_tensor(x, dtype=dtype)


class FilterEvaluator:
    """Evaluation metrics for accuracy, uncertainty calibration, and stability."""

    @staticmethod
    def compute_rmse(true_states: Any, particles: Any, log_weights: Any) -> tf.Tensor:
        """Compute RMSE between true states and weighted posterior mean trajectory."""
        true_states = _to_tensor(true_states)
        particles = _to_tensor(particles)
        log_weights = _to_tensor(log_weights)

        weights = tf.exp(log_weights)[..., tf.newaxis]
        estimates = tf.reduce_sum(particles * weights, axis=1)
        sq_err = tf.reduce_sum(tf.square(estimates - true_states), axis=-1)
        return tf.sqrt(tf.reduce_mean(sq_err))

    @staticmethod
    def compute_coverage(true_states: Any, particles: Any, alpha: float = 0.05) -> tf.Tensor:
        """Compute empirical coverage of central credible intervals from particles."""
        true_states = _to_tensor(true_states)
        particles = _to_tensor(particles)

        q_lo = 100.0 * (alpha / 2.0)
        q_hi = 100.0 * (1.0 - alpha / 2.0)
        ci = tfp.stats.percentile(particles, q=[q_lo, q_hi], axis=1)
        lo, hi = ci[0], ci[1]
        in_ci = tf.logical_and(true_states >= lo, true_states <= hi)
        return tf.reduce_mean(tf.cast(in_ci, tf.float32))

    @staticmethod
    def compute_ess(log_weights: Any) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute mean ESS and ESS trajectory from normalized log-weights."""
        log_weights = _to_tensor(log_weights)
        ess_traj = tf.exp(-tf.reduce_logsumexp(2.0 * log_weights, axis=1))
        return tf.reduce_mean(ess_traj), ess_traj

    @staticmethod
    def compute_rmse_percent(true_states: Any, particles: Any, log_weights: Any) -> tf.Tensor:
        """Compute bounded NRMSE percentage in [0, 100] using state-range normalization."""
        true_states = _to_tensor(true_states)
        rmse = FilterEvaluator.compute_rmse(true_states, particles, log_weights)
        state_range = tf.reduce_max(true_states) - tf.reduce_min(true_states)
        nrmse_pct = 100.0 * rmse / tf.maximum(state_range, 1e-8)
        return tf.clip_by_value(nrmse_pct, 0.0, 100.0)

    @staticmethod
    def compute_coverage_percent(true_states: Any, particles: Any, alpha: float = 0.05) -> tf.Tensor:
        """Compute empirical coverage in percentage scale [0, 100]."""
        cov = FilterEvaluator.compute_coverage(true_states, particles, alpha=alpha)
        return 100.0 * cov


class EfficiencyProfiler:
    """Context manager for wall-clock runtime and optional GPU peak memory."""

    def __init__(self):
        self.runtime = 0.0
        self.peak_memory_mb = 0.0
        self.gpus = tf.config.list_physical_devices("GPU")

    def __enter__(self):
        """Start timer and reset GPU memory stats when available."""
        if self.gpus:
            try:
                tf.config.experimental.reset_memory_stats("GPU:0")
            except (ValueError, RuntimeError):
                pass
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize runtime and capture peak GPU memory if supported."""
        del exc_type, exc_val, exc_tb
        self.runtime = time.time() - self.start_time
        if self.gpus:
            try:
                mem_info = tf.config.experimental.get_memory_info("GPU:0")
                self.peak_memory_mb = mem_info["peak"] / (1024**2)
            except (ValueError, AttributeError, RuntimeError):
                self.peak_memory_mb = -1.0


def evaluate_filter_run(
    true_states: Any,
    run_output: Dict[str, Any],
    alpha: float = 0.05,
    runtime_sec: float | None = None,
    peak_memory_mb: float | None = None,
) -> Dict[str, float]:
    """Evaluate one filter run and return scalar metrics.

    Args:
        true_states: Array-like of shape (T, state_dim).
        run_output: Dict containing at least `particles` and `log_weights`.
        alpha: Credible interval error level for coverage.

    Returns:
        Dict with RMSE (raw + percent), coverage (raw + percent), ESS,
        and optional runtime/peak-memory metrics.
    """
    particles = run_output["particles"]
    log_weights = run_output["log_weights"]

    rmse = FilterEvaluator.compute_rmse(true_states, particles, log_weights)
    rmse_percent = FilterEvaluator.compute_rmse_percent(true_states, particles, log_weights)
    coverage = FilterEvaluator.compute_coverage(true_states, particles, alpha=alpha)
    coverage_percent = FilterEvaluator.compute_coverage_percent(true_states, particles, alpha=alpha)
    mean_ess, ess_traj = FilterEvaluator.compute_ess(log_weights)

    metrics = {
        "rmse": float(rmse.numpy()),
        "rmse_percent": float(rmse_percent.numpy()),
        "coverage": float(coverage.numpy()),
        "coverage_percent": float(coverage_percent.numpy()),
        "mean_ess": float(mean_ess.numpy()),
        "final_ess": float(ess_traj[-1].numpy()),
    }
    if runtime_sec is not None:
        metrics["runtime_sec"] = float(runtime_sec)
    if peak_memory_mb is not None:
        metrics["peak_memory_mb"] = float(peak_memory_mb)
    return metrics