"""Compare DPFNet-PMMH vs DPFNet-HMC across scenarios.

This script evaluates three aspects:
1) Differentiability-bias trade-off
2) OT regularization effects
3) Gradient stability and variance

Outputs (under results/):
- pmmh_hmc_differentiability_bias_per_seed.csv
- pmmh_hmc_differentiability_bias_summary.csv
- pmmh_hmc_ot_regularization_per_seed.csv
- pmmh_hmc_ot_regularization_summary.csv
- pmmh_hmc_gradient_stability_per_seed.csv
- pmmh_hmc_gradient_stability_summary.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import build_registry, default_config, evaluate_filter_run, load_or_generate_data, run_filter_sequence
from utils.evaluate import EfficiencyProfiler
from utils.stability_diagnostics import run_stability_diagnostics

MODELS = ["DPFNet-HMC", "DPFNet-PMMH"]
OT_REG_GRID = [0.15, 0.35, 0.55]
EXTREME_SCENARIOS = [
    {
        "id": "extreme_sigmaV2_0p2_sigmaW2_60",
        "process_var": 0.2,
        "obs_var": 60.0,
        "dataset_path": "dataset/extreme_v0p2_w60.npz",
        "pmmh_proposal_scale": 6.0,
        "pmmh_inner_samples": 1,
        "pmmh_steps": 1,
        "pmmh_likelihood_jitter": 0.6,
    },
    {
        "id": "extreme_sigmaV2_60_sigmaW2_0p2",
        "process_var": 60.0,
        "obs_var": 0.2,
        "dataset_path": "dataset/extreme_v60_w0p2.npz",
        "pmmh_proposal_scale": 8.0,
        "pmmh_inner_samples": 1,
        "pmmh_steps": 1,
        "pmmh_likelihood_jitter": 0.8,
    },
    {
        "id": "extreme_sigmaV2_80_sigmaW2_80",
        "process_var": 80.0,
        "obs_var": 80.0,
        "dataset_path": "dataset/extreme_v80_w80.npz",
        "pmmh_proposal_scale": 10.0,
        "pmmh_inner_samples": 1,
        "pmmh_steps": 1,
        "pmmh_likelihood_jitter": 1.0,
    },
]


def _load_class(file_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(file_path.stem.replace("-", "_"), str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _build_model(cfg: dict, scenario: dict, model_name: str, transport_strength: float | None = None):
    common = {
        "state_dim": int(cfg["state_dim"]),
        "num_particles": int(cfg["num_particles"]),
        "process_var": float(scenario["process_var"]),
        "obs_var": float(scenario["obs_var"]),
        "init_var": float(cfg["init_var"]),
        "hidden_units": int(cfg.get("gradnet_hidden", 64)),
    }
    if transport_strength is not None:
        common["transport_strength"] = float(transport_strength)

    if model_name == "DPFNet-HMC":
        cls = _load_class((ROOT / "DPFNet-HMC/DPFNet-HMC.py").resolve(), "DPFNet_HMC")
        return cls(
            **common,
            hmc_steps=int(cfg.get("hmc_steps", 3)),
            hmc_leapfrog_steps=int(cfg.get("hmc_leapfrog_steps", 3)),
            hmc_step_size=float(cfg.get("hmc_step_size", 0.02)),
        )

    if model_name == "DPFNet-PMMH":
        cls = _load_class((ROOT / "DPFNet-PMMH/DPFNet-PMMH.py").resolve(), "DPFNet_PMMH")
        proposal_scale = float(scenario.get("pmmh_proposal_scale", 1.0))
        inner_samples = int(scenario.get("pmmh_inner_samples", int(cfg.get("pmmh_inner_samples", 4))))
        pmmh_steps = int(scenario.get("pmmh_steps", int(cfg.get("pmmh_steps", 3))))
        pmmh_likelihood_jitter = float(scenario.get("pmmh_likelihood_jitter", float(cfg.get("pmmh_likelihood_jitter", 0.1))))
        return cls(
            **common,
            pmmh_steps=max(1, pmmh_steps),
            pmmh_proposal_std=float(cfg.get("pmmh_proposal_std", 0.02)) * proposal_scale,
            pmmh_inner_samples=max(1, inner_samples),
            pmmh_likelihood_jitter=pmmh_likelihood_jitter,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _normalize_logw(logw: tf.Tensor) -> tf.Tensor:
    return logw - tf.reduce_logsumexp(logw)


def _analysis_config(cfg: dict, include_extreme: bool = True) -> dict:
    new_cfg = copy.deepcopy(cfg)
    if not include_extreme:
        return new_cfg

    existing = {str(s["id"]) for s in new_cfg.get("scenarios", [])}
    scenarios = list(new_cfg.get("scenarios", []))
    for scenario in EXTREME_SCENARIOS:
        if str(scenario["id"]) not in existing:
            scenarios.append(copy.deepcopy(scenario))
    new_cfg["scenarios"] = scenarios
    return new_cfg


def _build_two_model_registry(cfg: dict, scenario: dict, baseline_dir: Path) -> dict:
    registry_all = build_registry(cfg=cfg, scenario=scenario, baseline_dir=baseline_dir)
    return {k: v for k, v in registry_all.items() if k in set(MODELS)}


def _collect_stability_matrix(cfg: dict, workspace_root: Path) -> list[dict[str, Any]]:
    return run_stability_diagnostics(
        cfg=cfg,
        workspace_root=workspace_root,
        build_registry_fn=_build_two_model_registry,
        load_or_generate_data_fn=load_or_generate_data,
    )


def _transport_particles(model: Any, particles: tf.Tensor, logw: tf.Tensor) -> tf.Tensor:
    weights = tf.exp(logw)
    context = tf.concat([particles, weights[:, tf.newaxis]], axis=-1)
    net_scale = tf.cast(getattr(model, "net_scale", 0.04), particles.dtype)
    net_disp = net_scale * tf.tanh(model.grad_net(context))
    weighted_mean = tf.reduce_sum(particles * weights[:, tf.newaxis], axis=0, keepdims=True)
    pull_disp = tf.cast(model.transport_strength, particles.dtype) * (weighted_mean - particles)
    return particles + pull_disp + net_disp


def _posterior_bias(states: np.ndarray, run_output: dict[str, Any]) -> float:
    particles = tf.convert_to_tensor(run_output["particles"], dtype=tf.float32)
    logw = tf.convert_to_tensor(run_output["log_weights"], dtype=tf.float32)
    true_states = tf.convert_to_tensor(states, dtype=tf.float32)

    weights = tf.exp(logw)[..., tf.newaxis]
    estimate = tf.reduce_sum(particles * weights, axis=1)
    bias = tf.reduce_mean(tf.norm(estimate - true_states, axis=-1))
    return float(bias.numpy())


def _compute_local_differentiability(model: Any, particles: tf.Tensor, observation: tf.Tensor) -> float:
    eps = tf.constant(1e-3, dtype=particles.dtype)
    grad = model._grad_log_prob(particles, observation)
    grad_perturbed = model._grad_log_prob(particles, observation + eps * tf.ones_like(observation))
    sensitivity = tf.reduce_mean(tf.norm(grad_perturbed - grad, axis=-1)) / eps
    return float(sensitivity.numpy())


def _collect_differentiability_bias(
    cfg: dict,
    workspace_root: Path,
    stability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = [int(s) for s in cfg["experiment_seeds"]]
    stab_map = {
        (str(r["scenario"]), str(r["baseline"]), int(r["seed"])): r for r in stability_rows
    }

    for scenario in cfg["scenarios"]:
        scenario_id = str(scenario["id"])
        dataset_path = workspace_root / scenario["dataset_path"]
        states, observations = load_or_generate_data(
            dataset_path=dataset_path,
            cfg=cfg,
            process_var=float(scenario["process_var"]),
            obs_var=float(scenario["obs_var"]),
        )
        obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)
        time_steps = int(observations.shape[0])

        for model_name in MODELS:
            for seed in seeds:
                model = _build_model(cfg=cfg, scenario=scenario, model_name=model_name)
                output = run_filter_sequence(filter_obj=model, observations=obs_tf, seed=int(seed))
                metrics = evaluate_filter_run(
                    true_states=states,
                    run_output=output,
                    alpha=float(cfg["coverage_alpha"]),
                )
                bias = float(metrics["rmse"])
                stab = stab_map.get((scenario_id, model_name, int(seed)), {})
                jac_cond = float(stab.get("jacobian_conditioning_mean", np.nan))
                flow_std = float(stab.get("flow_magnitude_std", np.nan))
                differentiability = float(1.0 / (1.0 + max(flow_std, 1e-8))) if np.isfinite(flow_std) else np.nan

                acceptance_mean = float(tf.reduce_mean(tf.convert_to_tensor(output["resampled"], dtype=tf.float32)).numpy())
                rows.append(
                    {
                        "scenario": scenario_id,
                        "baseline": model_name,
                        "seed": int(seed),
                        "differentiability_sensitivity": differentiability,
                        "bias_l2": bias,
                        "diff_bias_product": float(differentiability * bias),
                        "flow_magnitude_std": flow_std,
                        "jacobian_conditioning": jac_cond,
                        "acceptance_or_resample_mean": acceptance_mean,
                    }
                )

    return rows


def _collect_ot_regularization_effects(cfg: dict, workspace_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = [int(s) for s in cfg["experiment_seeds"]]

    for scenario in cfg["scenarios"]:
        scenario_id = str(scenario["id"])
        dataset_path = workspace_root / scenario["dataset_path"]
        states, observations = load_or_generate_data(
            dataset_path=dataset_path,
            cfg=cfg,
            process_var=float(scenario["process_var"]),
            obs_var=float(scenario["obs_var"]),
        )
        obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)

        for model_name in MODELS:
            for reg in OT_REG_GRID:
                for seed in seeds:
                    model = _build_model(cfg=cfg, scenario=scenario, model_name=model_name, transport_strength=float(reg))
                    with EfficiencyProfiler() as profiler:
                        output = run_filter_sequence(filter_obj=model, observations=obs_tf, seed=int(seed))
                    metrics = evaluate_filter_run(
                        true_states=states,
                        run_output=output,
                        alpha=float(cfg["coverage_alpha"]),
                        runtime_sec=profiler.runtime,
                        peak_memory_mb=profiler.peak_memory_mb,
                    )
                    rows.append(
                        {
                            "scenario": scenario_id,
                            "baseline": model_name,
                            "seed": int(seed),
                            "transport_strength": float(reg),
                            "rmse": float(metrics["rmse"]),
                            "mean_ess": float(metrics["mean_ess"]),
                            "runtime_sec": float(metrics["runtime_sec"]),
                            "acceptance_or_resample_mean": float(
                                tf.reduce_mean(tf.convert_to_tensor(output["resampled"], dtype=tf.float32)).numpy()
                            ),
                        }
                    )

    return rows


def _collect_gradient_stability_variance(
    cfg: dict,
    workspace_root: Path,
    stability_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    del workspace_root
    for row in stability_rows:
        flow_mean = float(row["flow_magnitude_mean"])
        flow_std = float(row["flow_magnitude_std"])
        jac_mean = float(row["jacobian_conditioning_mean"])
        jac_std = float(row["jacobian_conditioning_std"])
        rows.append(
            {
                "scenario": str(row["scenario"]),
                "baseline": str(row["baseline"]),
                "seed": int(row["seed"]),
                "grad_norm_mean": flow_mean,
                "grad_norm_std": flow_std,
                "grad_norm_cv": float(flow_std / max(flow_mean, 1e-8)),
                "grad_delta_std": jac_std,
                "particle_var_mean": jac_mean,
                "particle_var_std": jac_std,
            }
        )

    return rows


def _summarize(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    value_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, vals in grouped.items():
        out = {k: v for k, v in zip(group_keys, key)}
        for metric in value_keys:
            arr = np.array([float(v[metric]) for v in vals], dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(arr))
            out[f"{metric}_std"] = float(np.std(arr, ddof=0)) if len(arr) > 1 else 0.0
        summary.append(out)

    summary.sort(key=lambda r: tuple(r[k] for k in group_keys))
    return summary


def _save_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_delta_summary(ot_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for row in ot_summary:
        key = (str(row["scenario"]), float(row["transport_strength"]))
        grouped.setdefault(key, {})[str(row["baseline"])] = row

    rows: list[dict[str, Any]] = []
    for (scenario, reg), models in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        if "DPFNet-HMC" not in models or "DPFNet-PMMH" not in models:
            continue
        hmc = models["DPFNet-HMC"]
        pmmh = models["DPFNet-PMMH"]

        rmse_hmc = float(hmc["rmse_mean"])
        rmse_pmmh = float(pmmh["rmse_mean"])
        diff = rmse_pmmh - rmse_hmc
        rel_pct = 100.0 * diff / max(abs(rmse_hmc), 1e-8)

        winner = "HMC" if diff > 0 else "PMMH"
        rows.append(
            {
                "scenario": scenario,
                "transport_strength": float(reg),
                "rmse_hmc": rmse_hmc,
                "rmse_pmmh": rmse_pmmh,
                "rmse_delta_pmmh_minus_hmc": float(diff),
                "rmse_relative_delta_percent": float(rel_pct),
                "acceptance_hmc": float(hmc["acceptance_or_resample_mean_mean"]),
                "acceptance_pmmh": float(pmmh["acceptance_or_resample_mean_mean"]),
                "quality_winner": winner,
            }
        )
    return rows


def _plot_scenario_comparison(
    diff_summary: list[dict[str, Any]],
    ot_summary: list[dict[str, Any]],
    grad_summary: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    scenarios = sorted({str(r["scenario"]) for r in diff_summary})
    model_order = ["DPFNet-HMC", "DPFNet-PMMH"]
    model_labels = {"DPFNet-HMC": "DPFNet-HMC", "DPFNet-PMMH": "DPFNet-PMMH"}

    for scenario in scenarios:
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
        axes_arr = axes if isinstance(axes, np.ndarray) else np.array([axes], dtype=object)

        diff_rows = [r for r in diff_summary if str(r["scenario"]) == scenario]
        grad_rows = [r for r in grad_summary if str(r["scenario"]) == scenario]
        ot_rows = [r for r in ot_summary if str(r["scenario"]) == scenario]

        x = np.arange(len(model_order), dtype=np.float32)
        width = 0.35

        diff_vals = [
            float(next((r["differentiability_sensitivity_mean"] for r in diff_rows if r["baseline"] == m), np.nan))
            for m in model_order
        ]
        bias_vals = [float(next((r["bias_l2_mean"] for r in diff_rows if r["baseline"] == m), np.nan)) for m in model_order]

        ax0 = axes_arr[0]
        ax0.bar(x - width / 2.0, diff_vals, width=width, label="Differentiability sensitivity")
        ax0.set_ylabel("Differentiability sensitivity")
        ax0.set_xticks(x)
        ax0.set_xticklabels([model_labels[m] for m in model_order], rotation=15, ha="right")
        ax0.grid(axis="y", alpha=0.25)
        ax0_t = ax0.twinx()
        ax0_t.bar(x + width / 2.0, bias_vals, width=width, alpha=0.55, color="tab:orange", label="Bias L2")
        ax0_t.set_ylabel("Bias L2")
        h0, l0 = ax0.get_legend_handles_labels()
        h1, l1 = ax0_t.get_legend_handles_labels()
        ax0.legend(h0 + h1, l0 + l1, fontsize=8, loc="upper left")
        ax0.set_title("Differentiability vs Bias")

        ax1 = axes_arr[1]
        for model_name in model_order:
            rows_m = [r for r in ot_rows if r["baseline"] == model_name]
            rows_m = sorted(rows_m, key=lambda r: float(r["transport_strength"]))
            regs = np.array([float(r["transport_strength"]) for r in rows_m], dtype=np.float64)
            rmse = np.array([float(r["rmse_mean"]) for r in rows_m], dtype=np.float64)
            ess = np.array([float(r["mean_ess_mean"]) for r in rows_m], dtype=np.float64)
            if regs.size > 0:
                ax1.plot(regs, rmse, marker="o", label=f"{model_labels[model_name]} RMSE")
                ax1.plot(regs, ess, marker="x", linestyle="--", alpha=0.75, label=f"{model_labels[model_name]} ESS")
        ax1.set_xlabel("Transport strength")
        ax1.set_ylabel("Metric value")
        ax1.set_title("OT Regularization Effects")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=7)

        cv_vals = [float(next((r["grad_norm_cv_mean"] for r in grad_rows if r["baseline"] == m), np.nan)) for m in model_order]
        delta_vals = [float(next((r["grad_delta_std_mean"] for r in grad_rows if r["baseline"] == m), np.nan)) for m in model_order]

        ax2 = axes_arr[2]
        ax2.bar(x - width / 2.0, cv_vals, width=width, label="Grad norm CV")
        ax2.bar(x + width / 2.0, delta_vals, width=width, alpha=0.7, label="Grad delta std")
        ax2.set_xticks(x)
        ax2.set_xticklabels([model_labels[m] for m in model_order], rotation=15, ha="right")
        ax2.set_ylabel("Stability metric")
        ax2.set_title("Gradient Stability & Variance")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=8)

        fig.suptitle(f"DPFNet-PMMH vs DPFNet-HMC | {scenario}")
        fig.tight_layout()
        out_path = out_dir / f"pmmh_hmc_comparison_{scenario}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DPFNet-PMMH and DPFNet-HMC across scenarios.")
    parser.add_argument("--config", type=str, default="", help="Optional path to JSON config overriding run.py defaults.")
    parser.add_argument(
        "--no-extreme",
        action="store_true",
        help="Disable added extreme scenarios (enabled by default).",
    )
    args = parser.parse_args()

    cfg = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8-sig") as f:
            cfg.update(json.load(f))
    cfg = _analysis_config(cfg=cfg, include_extreme=not args.no_extreme)

    tf.random.set_seed(int(cfg["global_tf_seed"]))
    np.random.seed(int(cfg["global_np_seed"]))

    stability_rows = _collect_stability_matrix(cfg=cfg, workspace_root=ROOT)

    diff_rows = _collect_differentiability_bias(cfg=cfg, workspace_root=ROOT, stability_rows=stability_rows)
    diff_summary = _summarize(
        rows=diff_rows,
        group_keys=("scenario", "baseline"),
        value_keys=(
            "differentiability_sensitivity",
            "bias_l2",
            "diff_bias_product",
            "flow_magnitude_std",
            "jacobian_conditioning",
            "acceptance_or_resample_mean",
        ),
    )

    ot_rows = _collect_ot_regularization_effects(cfg=cfg, workspace_root=ROOT)
    ot_summary = _summarize(
        rows=ot_rows,
        group_keys=("scenario", "baseline", "transport_strength"),
        value_keys=("rmse", "mean_ess", "runtime_sec", "acceptance_or_resample_mean"),
    )

    grad_rows = _collect_gradient_stability_variance(cfg=cfg, workspace_root=ROOT, stability_rows=stability_rows)
    grad_summary = _summarize(
        rows=grad_rows,
        group_keys=("scenario", "baseline"),
        value_keys=("grad_norm_mean", "grad_norm_cv", "grad_delta_std", "particle_var_mean"),
    )

    results_dir = ROOT / "results"
    _save_csv(diff_rows, results_dir / "pmmh_hmc_differentiability_bias_per_seed.csv")
    _save_csv(diff_summary, results_dir / "pmmh_hmc_differentiability_bias_summary.csv")
    _save_csv(ot_rows, results_dir / "pmmh_hmc_ot_regularization_per_seed.csv")
    _save_csv(ot_summary, results_dir / "pmmh_hmc_ot_regularization_summary.csv")
    _save_csv(grad_rows, results_dir / "pmmh_hmc_gradient_stability_per_seed.csv")
    _save_csv(grad_summary, results_dir / "pmmh_hmc_gradient_stability_summary.csv")
    delta_rows = _build_delta_summary(ot_summary=ot_summary)
    _save_csv(delta_rows, results_dir / "pmmh_hmc_extreme_delta_summary.csv")

    figure_paths = _plot_scenario_comparison(
        diff_summary=diff_summary,
        ot_summary=ot_summary,
        grad_summary=grad_summary,
        out_dir=ROOT / "figures",
    )

    print(f"Saved: {results_dir / 'pmmh_hmc_differentiability_bias_per_seed.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_differentiability_bias_summary.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_ot_regularization_per_seed.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_ot_regularization_summary.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_gradient_stability_per_seed.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_gradient_stability_summary.csv'}")
    print(f"Saved: {results_dir / 'pmmh_hmc_extreme_delta_summary.csv'}")
    for path in figure_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
