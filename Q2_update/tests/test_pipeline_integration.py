"""Integration tests for end-to-end filtering pipeline behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset.generate_dataset import generate_dataset
from run import run_filter_sequence
from utils.evaluate import evaluate_filter_run


def _load_baseline(file_name: str, class_name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "baselines" / file_name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _small_problem():
    states, observations = generate_dataset(
        seed=31,
        time_steps=12,
        state_dim=4,
        process_var=3.0,
        obs_var=1.0,
        init_var=5.0,
    )
    return states, observations


def test_runner_output_shapes_and_finiteness() -> None:
    """Runner must return correctly-shaped finite trajectories for particles and weights."""
    _, observations = _small_problem()
    DPFHS = _load_baseline("DFPHS.py", "DPFHS")
    filt = DPFHS(state_dim=4, num_particles=64, process_var=3.0, obs_var=1.0, init_var=5.0, ess_ratio=0.6)
    out = run_filter_sequence(filt, tf.convert_to_tensor(observations, dtype=tf.float32), seed=99)

    assert out["particles"].shape == (12, 64, 4)
    assert out["log_weights"].shape == (12, 64)
    assert out["ess"].shape == (12,)
    assert np.all(np.isfinite(out["particles"].numpy()))
    assert np.all(np.isfinite(out["log_weights"].numpy()))


def test_metrics_are_finite_and_reasonable() -> None:
    """Metrics from a short run should be finite and satisfy basic plausibility bounds."""
    states, observations = _small_problem()
    DPFS = _load_baseline("DPFS.py", "DPFS")
    filt = DPFS(state_dim=4, num_particles=64, process_var=3.0, obs_var=1.0, init_var=5.0, alpha=0.7)
    out = run_filter_sequence(filt, tf.convert_to_tensor(observations, dtype=tf.float32), seed=123)
    metrics = evaluate_filter_run(states, out, alpha=0.05)

    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["coverage"])
    assert np.isfinite(metrics["mean_ess"])
    assert metrics["rmse"] >= 0.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 1.0 <= metrics["mean_ess"] <= 64.0 + 1e-5
