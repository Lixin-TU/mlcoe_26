"""Stability diagnostics for particle-flow style model updates.

This module provides model-agnostic diagnostics over a common benchmark run:
- Flow magnitude: mean L2 displacement between pre-flow and post-flow particles.
- Jacobian conditioning: condition number of a linearized flow map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


IDENTITY_FLOW_MODELS = {"DFPHS", "DPFS"}


def _to_float_tensor(x: Any) -> tf.Tensor:
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
    return logw - tf.reduce_logsumexp(logw)


def _mean_flow_magnitude(x_before: tf.Tensor, x_after: tf.Tensor) -> float:
    disp = x_after - x_before
    mag = tf.reduce_mean(tf.norm(disp, axis=-1))
    return float(mag.numpy())


def _jacobian_condition_number(x_before: tf.Tensor, x_after: tf.Tensor, ridge: float = 1e-6) -> float:
    x_center = x_before - tf.reduce_mean(x_before, axis=0, keepdims=True)
    y_center = x_after - tf.reduce_mean(x_after, axis=0, keepdims=True)

    n = tf.cast(tf.shape(x_center)[0], tf.float32)
    d = tf.shape(x_center)[1]
    denom = tf.maximum(n - 1.0, 1.0)
    eye = tf.eye(d, dtype=tf.float32)

    cov_x = tf.matmul(x_center, x_center, transpose_a=True) / denom + tf.cast(ridge, tf.float32) * eye
    cross_xy = tf.matmul(x_center, y_center, transpose_a=True) / denom
    a = tf.linalg.solve(cov_x, cross_xy)

    singular_vals = tf.linalg.svd(a, compute_uv=False)
    s_max = singular_vals[0]
    s_min = tf.maximum(singular_vals[-1], tf.constant(1e-8, dtype=tf.float32))
    cond = s_max / s_min
    return float(cond.numpy())


def _transport_gradnet(model: Any, particles: tf.Tensor, logw: tf.Tensor) -> tf.Tensor:
    weights = tf.exp(logw)
    context = tf.concat([particles, weights[:, tf.newaxis]], axis=-1)
    if hasattr(model, "net_scale"):
        net_scale = tf.cast(model.net_scale, particles.dtype)
    else:
        net_scale = tf.cast(0.05, particles.dtype)
    net_disp = net_scale * tf.tanh(model.grad_net(context))
    weighted_mean = tf.reduce_sum(particles * weights[:, tf.newaxis], axis=0, keepdims=True)
    pull_disp = tf.cast(model.transport_strength, particles.dtype) * (weighted_mean - particles)
    return particles + pull_disp + net_disp


def _extract_flow_clouds(
    baseline_name: str,
    model: Any,
    particles: tf.Tensor,
    log_weights: tf.Tensor,
    observation: tf.Tensor,
    time_step: tf.Tensor,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    predicted = model._transition_sample(particles, time_step, seed=seed)
    logw = _normalize_logw(log_weights + model._observation_log_prob(predicted, observation))

    if baseline_name in IDENTITY_FLOW_MODELS:
        return predicted, predicted
    if baseline_name == "DPFOT":
        transported, _ = model._sinkhorn_transport(predicted, logw)
        return predicted, transported
    if baseline_name == "DPFOT-HMC":
        transported, _ = model._sinkhorn_transport(predicted, logw)
        return predicted, transported
    if baseline_name == "IPFPF":
        return predicted, model._flow_update(predicted, observation)
    if baseline_name == "SPFSM":
        return predicted, model._flow_update(predicted, observation, seed=seed)
    if baseline_name == "DPF-GradNet":
        return predicted, _transport_gradnet(model, predicted, logw)
    if baseline_name == "DPFNet-HMC":
        return predicted, _transport_gradnet(model, predicted, logw)
    if baseline_name == "DPFNet-PMMH":
        return predicted, _transport_gradnet(model, predicted, logw)

    return predicted, predicted


def run_stability_diagnostics(
    cfg: dict,
    workspace_root: Path,
    build_registry_fn,
    load_or_generate_data_fn,
) -> list[dict[str, Any]]:
    """Run stability diagnostics over all scenarios, seeds, and models.

    Returns per-(scenario, model, seed) aggregated diagnostics.
    """
    np.random.seed(int(cfg["global_np_seed"]))
    tf.random.set_seed(int(cfg["global_tf_seed"]))

    baseline_dir = workspace_root / "baselines"
    seeds = [int(s) for s in cfg["experiment_seeds"]]
    rows: list[dict[str, Any]] = []

    for scenario in cfg["scenarios"]:
        scenario_id = scenario["id"]
        dataset_path = workspace_root / scenario["dataset_path"]
        _, observations = load_or_generate_data_fn(
            dataset_path=dataset_path,
            cfg=cfg,
            process_var=float(scenario["process_var"]),
            obs_var=float(scenario["obs_var"]),
        )
        observations_tf = _to_float_tensor(observations)
        time_steps = int(observations_tf.shape[0])

        registry = build_registry_fn(cfg=cfg, scenario=scenario, baseline_dir=baseline_dir)
        for baseline_name, model in registry.items():
            for seed in seeds:
                particles, logw = model.initialize(seed=int(seed))
                flow_mags: list[float] = []
                jac_conds: list[float] = []

                for t in range(time_steps):
                    step_seed = int(seed) + t + 1
                    x_before, x_after = _extract_flow_clouds(
                        baseline_name=baseline_name,
                        model=model,
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )
                    flow_mags.append(_mean_flow_magnitude(x_before, x_after))
                    jac_conds.append(_jacobian_condition_number(x_before, x_after))

                    particles, logw, _, _ = model.step(
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )

                rows.append(
                    {
                        "scenario": scenario_id,
                        "baseline": baseline_name,
                        "seed": int(seed),
                        "flow_magnitude_mean": float(np.mean(flow_mags)),
                        "flow_magnitude_std": float(np.std(flow_mags, ddof=0)),
                        "jacobian_conditioning_mean": float(np.mean(jac_conds)),
                        "jacobian_conditioning_std": float(np.std(jac_conds, ddof=0)),
                    }
                )

    return rows


def summarize_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate diagnostics over scenarios and seeds by model."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for baseline, vals in grouped.items():
        flow_vals = np.array([float(v["flow_magnitude_mean"]) for v in vals], dtype=np.float64)
        jac_vals = np.array([float(v["jacobian_conditioning_mean"]) for v in vals], dtype=np.float64)
        summary.append(
            {
                "baseline": baseline,
                "flow_magnitude_mean": float(np.mean(flow_vals)),
                "flow_magnitude_std": float(np.std(flow_vals, ddof=0)),
                "jacobian_conditioning_mean": float(np.mean(jac_vals)),
                "jacobian_conditioning_std": float(np.std(jac_vals, ddof=0)),
            }
        )

    summary.sort(key=lambda x: x["baseline"])
    return summary


def run_stability_diagnostics_by_iteration(
    cfg: dict,
    workspace_root: Path,
    build_registry_fn,
    load_or_generate_data_fn,
) -> list[dict[str, Any]]:
    """Run diagnostics and aggregate flow metrics by model and iteration index.

    Iteration index matches filtering time-step index, i.e. t = 1..T.
    Aggregation pools all scenarios and seeds.
    """
    np.random.seed(int(cfg["global_np_seed"]))
    tf.random.set_seed(int(cfg["global_tf_seed"]))

    baseline_dir = workspace_root / "baselines"
    seeds = [int(s) for s in cfg["experiment_seeds"]]
    collector: dict[tuple[str, int], dict[str, list[float]]] = {}

    for scenario in cfg["scenarios"]:
        dataset_path = workspace_root / scenario["dataset_path"]
        _, observations = load_or_generate_data_fn(
            dataset_path=dataset_path,
            cfg=cfg,
            process_var=float(scenario["process_var"]),
            obs_var=float(scenario["obs_var"]),
        )
        observations_tf = _to_float_tensor(observations)
        time_steps = int(observations_tf.shape[0])

        registry = build_registry_fn(cfg=cfg, scenario=scenario, baseline_dir=baseline_dir)
        for baseline_name, model in registry.items():
            for seed in seeds:
                particles, logw = model.initialize(seed=int(seed))

                for t in range(time_steps):
                    step_seed = int(seed) + t + 1
                    x_before, x_after = _extract_flow_clouds(
                        baseline_name=baseline_name,
                        model=model,
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )

                    key = (baseline_name, t + 1)
                    if key not in collector:
                        collector[key] = {"flow": [], "jac": []}
                    collector[key]["flow"].append(_mean_flow_magnitude(x_before, x_after))
                    collector[key]["jac"].append(_jacobian_condition_number(x_before, x_after))

                    particles, logw, _, _ = model.step(
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )

    rows: list[dict[str, Any]] = []
    for (baseline, iteration), vals in collector.items():
        flow_vals = np.array(vals["flow"], dtype=np.float64)
        jac_vals = np.array(vals["jac"], dtype=np.float64)
        rows.append(
            {
                "baseline": baseline,
                "iteration": int(iteration),
                "flow_magnitude_mean": float(np.mean(flow_vals)),
                "flow_magnitude_std": float(np.std(flow_vals, ddof=0)),
                "jacobian_conditioning_mean": float(np.mean(jac_vals)),
                "jacobian_conditioning_std": float(np.std(jac_vals, ddof=0)),
            }
        )

    rows.sort(key=lambda x: (x["baseline"], int(x["iteration"])))
    return rows


def run_stability_diagnostics_by_iteration_scenario(
    cfg: dict,
    workspace_root: Path,
    build_registry_fn,
    load_or_generate_data_fn,
) -> list[dict[str, Any]]:
    """Run diagnostics aggregated by scenario, model, and iteration index."""
    np.random.seed(int(cfg["global_np_seed"]))
    tf.random.set_seed(int(cfg["global_tf_seed"]))

    baseline_dir = workspace_root / "baselines"
    seeds = [int(s) for s in cfg["experiment_seeds"]]
    collector: dict[tuple[str, str, int], dict[str, list[float]]] = {}

    for scenario in cfg["scenarios"]:
        scenario_id = str(scenario["id"])
        dataset_path = workspace_root / scenario["dataset_path"]
        _, observations = load_or_generate_data_fn(
            dataset_path=dataset_path,
            cfg=cfg,
            process_var=float(scenario["process_var"]),
            obs_var=float(scenario["obs_var"]),
        )
        observations_tf = _to_float_tensor(observations)
        time_steps = int(observations_tf.shape[0])

        registry = build_registry_fn(cfg=cfg, scenario=scenario, baseline_dir=baseline_dir)
        for baseline_name, model in registry.items():
            for seed in seeds:
                particles, logw = model.initialize(seed=int(seed))

                for t in range(time_steps):
                    step_seed = int(seed) + t + 1
                    x_before, x_after = _extract_flow_clouds(
                        baseline_name=baseline_name,
                        model=model,
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )

                    key = (scenario_id, baseline_name, t + 1)
                    if key not in collector:
                        collector[key] = {"flow": [], "jac": []}
                    collector[key]["flow"].append(_mean_flow_magnitude(x_before, x_after))
                    collector[key]["jac"].append(_jacobian_condition_number(x_before, x_after))

                    particles, logw, _, _ = model.step(
                        particles=particles,
                        log_weights=logw,
                        observation=observations_tf[t],
                        time_step=tf.cast(t + 1, tf.float32),
                        seed=step_seed,
                    )

    rows: list[dict[str, Any]] = []
    for (scenario_id, baseline, iteration), vals in collector.items():
        flow_vals = np.array(vals["flow"], dtype=np.float64)
        jac_vals = np.array(vals["jac"], dtype=np.float64)
        rows.append(
            {
                "scenario": scenario_id,
                "baseline": baseline,
                "iteration": int(iteration),
                "flow_magnitude_mean": float(np.mean(flow_vals)),
                "flow_magnitude_std": float(np.std(flow_vals, ddof=0)),
                "jacobian_conditioning_mean": float(np.mean(jac_vals)),
                "jacobian_conditioning_std": float(np.std(jac_vals, ddof=0)),
            }
        )

    rows.sort(key=lambda x: (str(x["scenario"]), str(x["baseline"]), int(x["iteration"])))
    return rows
