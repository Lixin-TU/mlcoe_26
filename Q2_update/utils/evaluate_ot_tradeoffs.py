"""Evaluate regularization-iterations-speed trade-offs for OT/Sinkhorn baselines.

Outputs:
- results/ot_tradeoff_per_seed.csv
- results/ot_tradeoff_summary.csv
- figures/ot_tradeoff_<model>_<scenario>.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import default_config, evaluate_filter_run, load_or_generate_data, run_filter_sequence
from utils.evaluate import EfficiencyProfiler


TARGET_MODELS = ["DPFOT", "DPFOT-HMC", "DPF-GradNet", "DPFNet-HMC"]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "DPFOT": {
        "module_path": "baselines/DPFOT.py",
        "class_name": "DPFOT",
        "reg_key": "epsilon",
        "iter_key": "sinkhorn_iters",
        "reg_grid": [0.05, 0.10, 0.20],
        "iter_grid": [20, 50, 100],
    },
    "DPFOT-HMC": {
        "module_path": "baselines/DPFOT-HMC.py",
        "class_name": "DPFOT_HMC",
        "reg_key": "epsilon",
        "iter_key": "sinkhorn_iters",
        "reg_grid": [0.06, 0.12, 0.24],
        "iter_grid": [20, 50, 100],
    },
    "DPF-GradNet": {
        "module_path": "baselines/DPF-GradNet.py",
        "class_name": "DPF_GradNet",
        "reg_key": "transport_strength",
        "iter_key": "transport_passes",
        "reg_grid": [0.20, 0.35, 0.50],
        "iter_grid": [1, 2, 4],
    },
    "DPFNet-HMC": {
        "module_path": "DPFNet-HMC/DPFNet-HMC.py",
        "class_name": "DPFNet_HMC",
        "reg_key": "transport_strength",
        "iter_key": "hmc_steps",
        "reg_grid": [0.20, 0.35, 0.50],
        "iter_grid": [1, 2, 4],
    },
}


def _load_class(file_path: Path, class_name: str):
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(file_path.stem.replace("-", "_"), str(file_path))
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)


class IteratedGradNet:
    """Lightweight wrapper to repeat GradNet transport passes per filtering step."""

    def __init__(self, base_model: Any, transport_passes: int = 1) -> None:
        self.base_model = base_model
        self.transport_passes = max(1, int(transport_passes))
        self.num_particles = int(base_model.num_particles)

    def initialize(self, seed: int):
        return self.base_model.initialize(seed=seed)

    def _transport_once(self, particles: tf.Tensor, logw: tf.Tensor) -> tf.Tensor:
        weights = tf.exp(logw)
        context = tf.concat([particles, weights[:, tf.newaxis]], axis=-1)
        net_disp = 0.05 * tf.tanh(self.base_model.grad_net(context))
        weighted_mean = tf.reduce_sum(particles * weights[:, tf.newaxis], axis=0, keepdims=True)
        pull_disp = tf.cast(self.base_model.transport_strength, particles.dtype) * (weighted_mean - particles)
        return particles + pull_disp + net_disp

    def step(self, particles: tf.Tensor, log_weights: tf.Tensor, observation: tf.Tensor, time_step: tf.Tensor, seed: int):
        m = self.base_model
        particles = m._transition_sample(particles, time_step, seed=seed)
        logw = m._normalize_logw(log_weights + m._observation_log_prob(particles, observation))
        ess = m._ess(logw)
        x_new = particles
        for _ in range(self.transport_passes):
            x_new = self._transport_once(x_new, logw)
        lw_new = tf.fill([self.num_particles], -tf.math.log(tf.cast(self.num_particles, tf.float32)))
        return x_new, lw_new, ess, tf.constant(1.0, dtype=tf.float32)


def _build_model(cfg: dict, scenario: dict, model_name: str, reg_value: float, iter_value: int):
    spec = MODEL_SPECS[model_name]
    cls = _load_class((ROOT / spec["module_path"]).resolve(), spec["class_name"])

    common_kwargs = {
        "state_dim": int(cfg["state_dim"]),
        "num_particles": int(cfg["num_particles"]),
        "process_var": float(scenario["process_var"]),
        "obs_var": float(scenario["obs_var"]),
        "init_var": float(cfg["init_var"]),
    }

    if model_name == "DPFOT":
        return cls(
            **common_kwargs,
            epsilon=float(reg_value),
            sinkhorn_iters=int(iter_value),
        )
    if model_name == "DPFOT-HMC":
        return cls(
            **common_kwargs,
            epsilon=float(reg_value),
            sinkhorn_iters=int(iter_value),
            hmc_steps=int(cfg["dpfot_hmc_steps"]),
            hmc_leapfrog_steps=int(cfg["dpfot_hmc_leapfrog_steps"]),
            hmc_step_size=float(cfg["dpfot_hmc_step_size"]),
        )
    if model_name == "DPF-GradNet":
        base = cls(
            **common_kwargs,
            hidden_units=int(cfg["gradnet_hidden"]),
            transport_strength=float(reg_value),
        )
        return IteratedGradNet(base_model=base, transport_passes=int(iter_value))
    if model_name == "DPFNet-HMC":
        return cls(
            **common_kwargs,
            hidden_units=int(cfg["gradnet_hidden"]),
            transport_strength=float(reg_value),
            hmc_steps=int(iter_value),
            hmc_leapfrog_steps=int(cfg["hmc_leapfrog_steps"]),
            hmc_step_size=float(cfg["hmc_step_size"]),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["scenario"]),
            str(row["baseline"]),
            float(row["regularization"]),
            int(row["iterations"]),
        )
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (scenario, baseline, reg, iterations), vals in grouped.items():
        rmse_vals = np.array([float(v["rmse"]) for v in vals], dtype=np.float64)
        ess_vals = np.array([float(v["mean_ess"]) for v in vals], dtype=np.float64)
        runtime_vals = np.array([float(v["runtime_sec"]) for v in vals], dtype=np.float64)
        out.append(
            {
                "scenario": scenario,
                "baseline": baseline,
                "regularization": float(reg),
                "iterations": int(iterations),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals, ddof=0)) if len(rmse_vals) > 1 else 0.0,
                "ess_mean": float(np.mean(ess_vals)),
                "ess_std": float(np.std(ess_vals, ddof=0)) if len(ess_vals) > 1 else 0.0,
                "runtime_sec_mean": float(np.mean(runtime_vals)),
                "runtime_sec_std": float(np.std(runtime_vals, ddof=0)) if len(runtime_vals) > 1 else 0.0,
            }
        )

    out.sort(key=lambda r: (r["scenario"], r["baseline"], r["regularization"], r["iterations"]))
    return out


def _pivot(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[float], list[int]]:
    reg_vals = sorted(df["regularization"].astype(float).unique().tolist())
    iter_vals = sorted(df["iterations"].astype(int).unique().tolist())
    table = np.full((len(reg_vals), len(iter_vals)), np.nan, dtype=np.float64)
    reg_to_i = {v: i for i, v in enumerate(reg_vals)}
    it_to_j = {v: j for j, v in enumerate(iter_vals)}
    for _, row in df.iterrows():
        i = reg_to_i[float(row["regularization"])]
        j = it_to_j[int(row["iterations"])]
        table[i, j] = float(row[value_col])
    return table, reg_vals, iter_vals


def _plot_heatmap_panels(summary_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    df_all = pd.DataFrame(summary_rows)
    if df_all.empty:
        return saved

    for scenario in sorted(df_all["scenario"].unique().tolist()):
        df_s = df_all[df_all["scenario"] == scenario]
        for model_name in TARGET_MODELS:
            df_m = df_s[df_s["baseline"] == model_name]
            if df_m.empty:
                continue

            rmse_grid, reg_vals, iter_vals = _pivot(df_m, "rmse_mean")
            ess_grid, _, _ = _pivot(df_m, "ess_mean")
            runtime_grid, _, _ = _pivot(df_m, "runtime_sec_mean")

            fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
            panels = [
                (rmse_grid, "RMSE (lower is better)", "viridis"),
                (ess_grid, "Mean ESS (higher is better)", "plasma"),
                (runtime_grid, "Runtime [s] (lower is better)", "magma"),
            ]

            for ax, (grid, title, cmap) in zip(axes, panels):
                im = ax.imshow(grid, aspect="auto", cmap=cmap)
                ax.set_title(title)
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Regularization")
                ax.set_xticks(np.arange(len(iter_vals)))
                ax.set_xticklabels([str(v) for v in iter_vals])
                ax.set_yticks(np.arange(len(reg_vals)))
                ax.set_yticklabels([f"{v:.3g}" for v in reg_vals])
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            display_name = "DPFNet-HMC (proposed)" if model_name == "DPFNet-HMC" else model_name
            fig.suptitle(f"OT Trade-off Analysis | {display_name} | {scenario}", y=1.02)
            fig.tight_layout()

            safe_model = model_name.replace("/", "-").replace(" ", "_")
            out_path = out_dir / f"ot_tradeoff_{safe_model}_{scenario}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    return saved


def _save_csv(rows: list[dict[str, Any]], out_csv: Path, fieldnames: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_tradeoff_analysis(cfg: dict, workspace_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    tf.random.set_seed(int(cfg["global_tf_seed"]))
    np.random.seed(int(cfg["global_np_seed"]))

    seeds = [int(s) for s in cfg["experiment_seeds"]]
    rows: list[dict[str, Any]] = []

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

        for model_name in TARGET_MODELS:
            spec = MODEL_SPECS[model_name]
            for reg_value in spec["reg_grid"]:
                for iter_value in spec["iter_grid"]:
                    for seed in seeds:
                        model = _build_model(
                            cfg=cfg,
                            scenario=scenario,
                            model_name=model_name,
                            reg_value=float(reg_value),
                            iter_value=int(iter_value),
                        )
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
                                "regularization": float(reg_value),
                                "iterations": int(iter_value),
                                "regularization_param": spec["reg_key"],
                                "iterations_param": spec["iter_key"],
                                "rmse": float(metrics["rmse"]),
                                "rmse_percent": float(metrics["rmse_percent"]),
                                "coverage": float(metrics["coverage"]),
                                "coverage_percent": float(metrics["coverage_percent"]),
                                "mean_ess": float(metrics["mean_ess"]),
                                "runtime_sec": float(metrics["runtime_sec"]),
                                "peak_memory_mb": float(metrics["peak_memory_mb"]),
                            }
                        )

    summary = _aggregate(rows)
    figures = _plot_heatmap_panels(summary_rows=summary, out_dir=workspace_root / "figures")
    return rows, summary, figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze regularization-iterations-speed trade-offs for OT/Sinkhorn baselines.")
    parser.add_argument("--config", type=str, default="", help="Optional path to JSON config overriding run.py defaults.")
    args = parser.parse_args()

    cfg = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8-sig") as f:
            cfg.update(json.load(f))

    rows, summary, fig_paths = run_tradeoff_analysis(cfg=cfg, workspace_root=ROOT)

    results_dir = ROOT / "results"
    _save_csv(
        rows=rows,
        out_csv=results_dir / "ot_tradeoff_per_seed.csv",
        fieldnames=[
            "scenario",
            "baseline",
            "seed",
            "regularization",
            "iterations",
            "regularization_param",
            "iterations_param",
            "rmse",
            "rmse_percent",
            "coverage",
            "coverage_percent",
            "mean_ess",
            "runtime_sec",
            "peak_memory_mb",
        ],
    )
    _save_csv(
        rows=summary,
        out_csv=results_dir / "ot_tradeoff_summary.csv",
        fieldnames=[
            "scenario",
            "baseline",
            "regularization",
            "iterations",
            "rmse_mean",
            "rmse_std",
            "ess_mean",
            "ess_std",
            "runtime_sec_mean",
            "runtime_sec_std",
        ],
    )

    print(f"Saved per-seed trade-off table: {results_dir / 'ot_tradeoff_per_seed.csv'}")
    print(f"Saved trade-off summary table: {results_dir / 'ot_tradeoff_summary.csv'}")
    for path in fig_paths:
        print(f"Saved trade-off figure: {path}")


if __name__ == "__main__":
    main()
