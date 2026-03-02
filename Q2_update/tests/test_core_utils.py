"""Unit tests for core utilities and deterministic data generation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from dataset.generate_dataset import generate_dataset
from utils.evaluate import FilterEvaluator


def test_generate_dataset_is_reproducible() -> None:
    """Dataset generation must be deterministic for the same seed/config."""
    x1, y1 = generate_dataset(seed=7, time_steps=20, state_dim=3, process_var=2.0, obs_var=1.0, init_var=5.0)
    x2, y2 = generate_dataset(seed=7, time_steps=20, state_dim=3, process_var=2.0, obs_var=1.0, init_var=5.0)
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_coverage_percent_bounds() -> None:
    """Coverage percentage should stay within [0, 100]."""
    true_states = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
    particles = tf.constant(
        [
            [[-1.0], [0.0], [1.0]],
            [[0.0], [1.0], [2.0]],
            [[1.0], [2.0], [3.0]],
        ],
        dtype=tf.float32,
    )
    coverage_pct = float(FilterEvaluator.compute_coverage_percent(true_states, particles, alpha=0.05).numpy())
    assert 0.0 <= coverage_pct <= 100.0


def test_ess_bounds_from_log_weights() -> None:
    """ESS computed from normalized log-weights must lie in [1, N]."""
    n = 50
    raw = tf.random.stateless_uniform((4, n), seed=(5, 6), minval=0.0, maxval=1.0)
    probs = raw / tf.reduce_sum(raw, axis=1, keepdims=True)
    logw = tf.math.log(probs + 1e-8)
    mean_ess, _ = FilterEvaluator.compute_ess(logw)
    ess = float(mean_ess.numpy())
    assert ess >= 1.0
    assert ess <= n + 1e-5


def test_rmse_percent_bounded() -> None:
    """RMSE percentage metric is clipped to [0, 100] by design."""
    true_states = tf.zeros((5, 2), dtype=tf.float32)
    particles = tf.ones((5, 10, 2), dtype=tf.float32) * 1000.0
    logw = tf.fill((5, 10), -tf.math.log(10.0))
    rmse_pct = float(FilterEvaluator.compute_rmse_percent(true_states, particles, logw).numpy())
    assert 0.0 <= rmse_pct <= 100.0
