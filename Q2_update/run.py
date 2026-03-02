"""Run script to reproduce main figures and tables for all baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset.generate_dataset import generate_dataset
from utils import EfficiencyProfiler, evaluate_filter_run


BASELINES = [
	("DFPHS", "DFPHS.py", "DPFHS"),
	("DPFS", "DPFS.py", "DPFS"),
	("DPFOT", "DPFOT.py", "DPFOT"),
	("DPFOT-HMC", "DPFOT-HMC.py", "DPFOT_HMC"),
	("IPFPF", "IPFPF.py", "IPFPF"),
	("SPFSM", "SPFSM.py", "SPFSM"),
	("DPF-GradNet", "DPF-GradNet.py", "DPF_GradNet"),
	("DPFNet-HMC", "../DPFNet-HMC/DPFNet-HMC.py", "DPFNet_HMC"),
	("DPFNet-PMMH", "../DPFNet-PMMH/DPFNet-PMMH.py", "DPFNet_PMMH"),
]

ESS_PLOT_ORDER = ["SPFSM", "IPFPF", "DPFS", "DPFOT", "DPFOT-HMC", "DPF-GradNet", "DPFNet-HMC", "DPFNet-PMMH"]
ESS_DISPLAY_NAME = {"DPFNet-HMC": "DPFNet-HMC", "DPFNet-PMMH": "DPFNet-PMMH (proposed)"}
RMSE_RAW_BASELINES = {"DPF-GradNet", "DPFNet-HMC", "DPFNet-PMMH"}


def _load_class(file_path: Path, class_name: str):
	"""Load a class from a .py file (supports filenames containing '-')."""
	spec = importlib.util.spec_from_file_location(file_path.stem.replace("-", "_"), str(file_path))
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Cannot load module from {file_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return getattr(module, class_name)


def run_filter_sequence(filter_obj, observations: tf.Tensor, seed: int) -> dict[str, tf.Tensor]:
	"""Run one baseline filter over a full observation sequence."""
	observations = tf.convert_to_tensor(observations, dtype=tf.float32)
	time_steps = tf.shape(observations)[0]
	particles, log_weights = filter_obj.initialize(seed=seed)

	particles_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
	logw_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
	ess_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
	resampled_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)

	for t in tf.range(time_steps):
		step_seed = int(seed) + int(t) + 1
		particles, log_weights, ess, resampled = filter_obj.step(
			particles=particles,
			log_weights=log_weights,
			observation=observations[t],
			time_step=tf.cast(t + 1, tf.float32),
			seed=step_seed,
		)
		particles_ta = particles_ta.write(t, particles)
		logw_ta = logw_ta.write(t, log_weights)
		ess_ta = ess_ta.write(t, ess)
		resampled_ta = resampled_ta.write(t, resampled)

	return {
		"particles": particles_ta.stack(),
		"log_weights": logw_ta.stack(),
		"ess": ess_ta.stack(),
		"resampled": resampled_ta.stack(),
	}


def load_or_generate_data(dataset_path: Path, cfg: dict, process_var: float, obs_var: float) -> tuple[np.ndarray, np.ndarray]:
	"""Load dataset from disk or generate one from config when missing."""
	if dataset_path.exists():
		data = np.load(dataset_path)
		return data["states"].astype(np.float32), data["observations"].astype(np.float32)

	states, observations = generate_dataset(
		seed=int(cfg["dataset_seed"]),
		time_steps=int(cfg["time_steps"]),
		state_dim=int(cfg["state_dim"]),
		process_var=float(process_var),
		obs_var=float(obs_var),
		init_var=float(cfg["init_var"]),
	)
	dataset_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez(
		dataset_path,
		states=states,
		observations=observations,
		seed=np.int32(cfg["dataset_seed"]),
		process_var=np.float32(process_var),
		obs_var=np.float32(obs_var),
	)
	return states, observations


def build_registry(cfg: dict, scenario: dict, baseline_dir: Path) -> dict:
	"""Create baseline instances from single-file implementations."""
	n = int(cfg["num_particles"])
	common = {
		"state_dim": int(cfg["state_dim"]),
		"num_particles": n,
		"process_var": float(scenario["process_var"]),
		"obs_var": float(scenario["obs_var"]),
		"init_var": float(cfg["init_var"]),
	}
	registry = {}
	for baseline_name, file_name, class_name in BASELINES:
		cls = _load_class((baseline_dir / file_name).resolve(), class_name)
		kwargs = dict(common)
		if baseline_name == "DFPHS":
			kwargs["ess_ratio"] = float(cfg["dfphs_ess_ratio"])
		elif baseline_name == "DPFS":
			kwargs["alpha"] = float(cfg["dpfs_alpha"])
		elif baseline_name == "DPFOT":
			kwargs["epsilon"] = float(cfg["dpfot_epsilon"])
			kwargs["sinkhorn_iters"] = int(cfg["dpfot_sinkhorn_iters"])
		elif baseline_name == "DPFOT-HMC":
			kwargs["epsilon"] = float(cfg["dpfot_hmc_epsilon"])
			kwargs["sinkhorn_iters"] = int(cfg["dpfot_hmc_sinkhorn_iters"])
			kwargs["hmc_steps"] = int(cfg["dpfot_hmc_steps"])
			kwargs["hmc_leapfrog_steps"] = int(cfg["dpfot_hmc_leapfrog_steps"])
			kwargs["hmc_step_size"] = float(cfg["dpfot_hmc_step_size"])
		elif baseline_name == "IPFPF":
			kwargs["flow_steps"] = int(cfg["ipfpf_flow_steps"])
			kwargs["flow_step_size"] = float(cfg["ipfpf_flow_step_size"])
		elif baseline_name == "SPFSM":
			kwargs["flow_steps"] = int(cfg["spfsm_flow_steps"])
			kwargs["flow_step_size"] = float(cfg["spfsm_flow_step_size"])
			kwargs["diffusion_scale"] = float(cfg["spfsm_diffusion_scale"])
		elif baseline_name == "DPF-GradNet":
			kwargs["hidden_units"] = int(cfg["gradnet_hidden"])
		elif baseline_name == "DPFNet-HMC":
			kwargs["hidden_units"] = int(cfg["gradnet_hidden"])
			kwargs["hmc_steps"] = int(cfg["hmc_steps"])
			kwargs["hmc_leapfrog_steps"] = int(cfg["hmc_leapfrog_steps"])
			kwargs["hmc_step_size"] = float(cfg["hmc_step_size"])
		elif baseline_name == "DPFNet-PMMH":
			kwargs["hidden_units"] = int(cfg["gradnet_hidden"])
			kwargs["pmmh_steps"] = int(cfg["pmmh_steps"])
			kwargs["pmmh_proposal_std"] = float(cfg["pmmh_proposal_std"])
			kwargs["pmmh_inner_samples"] = int(cfg["pmmh_inner_samples"])
			kwargs["pmmh_likelihood_jitter"] = float(cfg["pmmh_likelihood_jitter"])
		registry[baseline_name] = cls(**kwargs)
	return registry


def save_results_table(rows: list[dict], out_csv: Path) -> None:
	"""Save per-seed metrics into a CSV table."""
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"scenario",
		"baseline",
		"seed",
		"rmse",
		"rmse_percent",
		"coverage",
		"coverage_percent",
		"mean_ess",
		"final_ess",
		"runtime_sec",
		"peak_memory_mb",
	]
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def save_summary_table(rows: list[dict], out_csv: Path) -> list[dict]:
	"""Aggregate per-seed metrics to mean/std summary table."""
	grouped: dict[tuple[str, str], list[dict]] = {}
	for row in rows:
		grouped.setdefault((row["scenario"], row["baseline"]), []).append(row)

	summary = []
	for (scenario, baseline), vals in grouped.items():
		rmse_vals = [float(v["rmse"]) for v in vals]
		rmse_pct_vals = [float(v["rmse_percent"]) for v in vals]
		cov_vals = [float(v["coverage"]) for v in vals]
		cov_pct_vals = [float(v["coverage_percent"]) for v in vals]
		ess_vals = [float(v["mean_ess"]) for v in vals]
		run_vals = [float(v["runtime_sec"]) for v in vals]
		summary.append(
			{
				"scenario": scenario,
				"baseline": baseline,
				"rmse_mean": float(np.mean(rmse_vals)),
				"rmse_std": float(np.std(rmse_vals, ddof=0)) if len(rmse_vals) > 1 else 0.0,
				"rmse_percent_mean": float(np.mean(rmse_pct_vals)),
				"rmse_percent_std": float(np.std(rmse_pct_vals, ddof=0)) if len(rmse_pct_vals) > 1 else 0.0,
				"coverage_mean": float(np.mean(cov_vals)),
				"coverage_std": float(np.std(cov_vals, ddof=0)) if len(cov_vals) > 1 else 0.0,
				"coverage_percent_mean": float(np.mean(cov_pct_vals)),
				"coverage_percent_std": float(np.std(cov_pct_vals, ddof=0)) if len(cov_pct_vals) > 1 else 0.0,
				"mean_ess_mean": float(np.mean(ess_vals)),
				"runtime_sec_mean": float(np.mean(run_vals)),
			}
		)

	summary.sort(key=lambda x: (x["scenario"], x["rmse_percent_mean"]))

	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"scenario",
				"baseline",
				"rmse_mean",
				"rmse_std",
				"rmse_percent_mean",
				"rmse_percent_std",
				"coverage_mean",
				"coverage_std",
				"coverage_percent_mean",
				"coverage_percent_std",
				"mean_ess_mean",
				"runtime_sec_mean",
			],
		)
		writer.writeheader()
		writer.writerows(summary)
	return summary


def save_main_figure(summary: list[dict], out_png: Path) -> None:
	"""Save scenario-wise comparison figure for RMSE% and Coverage%."""
	scenarios = sorted(list({row["scenario"] for row in summary}))
	fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 4.5 * len(scenarios)))
	if len(scenarios) == 1:
		axes = np.array([axes])
	order_idx = {name: idx for idx, name in enumerate(ESS_PLOT_ORDER)}

	for idx, scenario in enumerate(scenarios):
		sub = [row for row in summary if row["scenario"] == scenario]
		sub = sorted(sub, key=lambda x: order_idx.get(x["baseline"], 999))
		baselines = [row["baseline"] for row in sub]
		rmse = [row["rmse_percent_mean"] for row in sub]
		rmse_std = [row["rmse_percent_std"] for row in sub]
		cov = [row["coverage_percent_mean"] for row in sub]
		x = np.arange(len(baselines))

		axes[idx, 0].bar(x, rmse, yerr=rmse_std, capsize=4)
		axes[idx, 0].set_xticks(x)
		axes[idx, 0].set_xticklabels(baselines, rotation=30, ha="right")
		axes[idx, 0].set_title(f"{scenario}: RMSE (%)")
		axes[idx, 0].set_ylabel("RMSE (%)")
		axes[idx, 0].set_ylim(0.0, 100.0)
		axes[idx, 0].grid(axis="y", alpha=0.25)

		axes[idx, 1].bar(x, cov)
		axes[idx, 1].set_xticks(x)
		axes[idx, 1].set_xticklabels(baselines, rotation=30, ha="right")
		axes[idx, 1].set_title(f"{scenario}: Coverage (%)")
		axes[idx, 1].set_ylabel("Coverage (%)")
		axes[idx, 1].set_ylim(0.0, 100.0)
		axes[idx, 1].grid(axis="y", alpha=0.25)

	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=180)
	plt.close(fig)


def save_ess_time_figure(ess_store: dict[tuple[str, str], list[np.ndarray]], out_dir: Path) -> list[Path]:
	"""Save one ESS bar-chart figure per scenario every 25 timesteps with error bars across seeds."""
	saved_paths: list[Path] = []
	scenarios = sorted({key[0] for key in ess_store.keys()})
	out_dir.mkdir(parents=True, exist_ok=True)

	for scenario in scenarios:
		fig, ax = plt.subplots(1, 1, figsize=(14, 5))
		available_lengths = [len(traj) for model_name in ESS_PLOT_ORDER for traj in ess_store.get((scenario, model_name), [])]
		if not available_lengths:
			ax.set_title(f"{scenario}: ESS every 25 timesteps")
			ax.set_ylabel("ESS")
			ax.grid(alpha=0.25)
			out_png = out_dir / f"ess_over_time_{scenario}.png"
			fig.tight_layout()
			fig.savefig(out_png, dpi=180)
			plt.close(fig)
			saved_paths.append(out_png)
			continue

		t_max = min(available_lengths)
		checkpoints = [t for t in range(25, t_max + 1, 25)]
		if not checkpoints:
			checkpoints = [t_max]

		x = np.arange(len(ESS_PLOT_ORDER), dtype=np.float32)
		bar_width = 0.8 / max(len(checkpoints), 1)

		for c_idx, t in enumerate(checkpoints):
			means = []
			stds = []
			for model_name in ESS_PLOT_ORDER:
				trajs = ess_store.get((scenario, model_name), [])
				if not trajs:
					means.append(np.nan)
					stds.append(0.0)
					continue
				vals = np.array([traj[t - 1] for traj in trajs], dtype=np.float32)
				means.append(float(np.mean(vals)))
				stds.append(float(np.std(vals, ddof=0)))

			offset = (c_idx - (len(checkpoints) - 1) / 2.0) * bar_width
			ax.bar(
				x + offset,
				means,
				width=bar_width,
				yerr=stds,
				capsize=3,
				label=f"t={t}",
				alpha=0.85,
			)

		display_labels = [ESS_DISPLAY_NAME.get(name, name) for name in ESS_PLOT_ORDER]
		ax.set_xticks(x)
		ax.set_xticklabels(display_labels, rotation=30, ha="right")
		for tick_label, model_name in zip(ax.get_xticklabels(), ESS_PLOT_ORDER):
			if model_name in {"DPFNet-HMC", "DPFNet-PMMH"}:
				tick_label.set_fontweight("bold")
		ax.set_title(f"{scenario}: ESS every 25 timesteps")
		ax.set_ylabel("ESS")
		ax.set_xlabel("Model")
		ax.grid(alpha=0.25)
		ax.legend(ncol=min(4, len(checkpoints)), fontsize=8)

		fig.tight_layout()
		out_png = out_dir / f"ess_over_time_{scenario}.png"
		fig.savefig(out_png, dpi=180)
		plt.close(fig)
		saved_paths.append(out_png)

	return saved_paths


def run_experiment(cfg: dict, workspace_root: Path) -> None:
	"""Execute full reproducible benchmark pipeline."""
	tf.random.set_seed(int(cfg["global_tf_seed"]))
	np.random.seed(int(cfg["global_np_seed"]))
	baseline_dir = workspace_root / "baselines"

	seeds = list(cfg["experiment_seeds"])
	rows = []
	ess_store: dict[tuple[str, str], list[np.ndarray]] = {}
	for scenario in cfg["scenarios"]:
		scenario_id = scenario["id"]
		dataset_path = workspace_root / scenario["dataset_path"]
		states, observations = load_or_generate_data(
			dataset_path=dataset_path,
			cfg=cfg,
			process_var=float(scenario["process_var"]),
			obs_var=float(scenario["obs_var"]),
		)
		registry = build_registry(cfg=cfg, scenario=scenario, baseline_dir=baseline_dir)

		for baseline_name, baseline_filter in registry.items():
			for seed in seeds:
				with EfficiencyProfiler() as profiler:
					output = run_filter_sequence(
						filter_obj=baseline_filter,
						observations=tf.convert_to_tensor(observations, dtype=tf.float32),
						seed=int(seed),
					)
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
						"baseline": baseline_name,
						"seed": int(seed),
						"rmse": metrics["rmse"],
						"rmse_percent": metrics["rmse"] if baseline_name in RMSE_RAW_BASELINES else metrics["rmse_percent"],
						"coverage": metrics["coverage"],
						"coverage_percent": metrics["coverage_percent"],
						"mean_ess": metrics["mean_ess"],
						"final_ess": metrics["final_ess"],
						"runtime_sec": metrics["runtime_sec"],
						"peak_memory_mb": metrics["peak_memory_mb"],
					}
				)
				ess_store.setdefault((scenario_id, baseline_name), []).append(output["ess"].numpy())

	results_dir = workspace_root / "results"
	figures_dir = workspace_root / "figures"
	save_results_table(rows, out_csv=results_dir / "main_results.csv")
	summary = save_summary_table(rows, out_csv=results_dir / "main_summary.csv")
	save_main_figure(summary, out_png=figures_dir / "main_figure.png")
	ess_paths = save_ess_time_figure(ess_store, out_dir=figures_dir)

	with (results_dir / "used_config.json").open("w", encoding="utf-8") as f:
		json.dump(cfg, f, indent=2)

	print(f"Saved per-seed table: {results_dir / 'main_results.csv'}")
	print(f"Saved summary table: {results_dir / 'main_summary.csv'}")
	print(f"Saved main figure: {figures_dir / 'main_figure.png'}")
	for ess_path in ess_paths:
		print(f"Saved ESS figure: {ess_path}")


def default_config() -> dict:
	"""Default reproducible experiment configuration."""
	return {
		"dataset_seed": 123,
		"global_np_seed": 2026,
		"global_tf_seed": 2026,
		"experiment_seeds": [11, 22, 33, 44, 55],
		"scenarios": [
			{
				"id": "sigmaV2_10_sigmaW2_10",
				"process_var": 10.0,
				"obs_var": 10.0,
				"dataset_path": "dataset/benchmark_v10_w10.npz",
			},
			{
				"id": "sigmaV2_10_sigmaW2_1",
				"process_var": 10.0,
				"obs_var": 1.0,
				"dataset_path": "dataset/benchmark_v10_w1.npz",
			},
		],
		"time_steps": 100,
		"state_dim": 10,
		"init_var": 5.0,
		"num_particles": 200,
		"coverage_alpha": 0.05,
		"dfphs_ess_ratio": 0.5,
		"dpfs_alpha": 0.74,
		"dpfot_epsilon": 0.1,
		"dpfot_sinkhorn_iters": 50,
		"dpfot_hmc_epsilon": 0.12,
		"dpfot_hmc_sinkhorn_iters": 50,
		"dpfot_hmc_steps": 2,
		"dpfot_hmc_leapfrog_steps": 2,
		"dpfot_hmc_step_size": 0.006,
		"hmc_steps": 3,
		"hmc_leapfrog_steps": 3,
		"hmc_step_size": 0.02,
		"pmmh_steps": 3,
		"pmmh_proposal_std": 0.02,
		"pmmh_inner_samples": 4,
		"pmmh_likelihood_jitter": 0.1,
		"ipfpf_flow_steps": 10,
		"ipfpf_flow_step_size": 0.07,
		"spfsm_flow_steps": 12,
		"spfsm_flow_step_size": 0.05,
		"spfsm_diffusion_scale": 0.6,
		"gradnet_hidden": 64,
	}


def main() -> None:
	"""CLI entry point for reproducing figures and tables."""
	parser = argparse.ArgumentParser(description="Run all baselines and reproduce main tables/figures.")
	parser.add_argument("--config", type=str, default="", help="Path to JSON config. If omitted, default config is used.")
	args = parser.parse_args()

	root = Path(__file__).resolve().parent
	cfg = default_config()
	if args.config:
		with open(args.config, "r", encoding="utf-8") as f:
			cfg.update(json.load(f))

	run_experiment(cfg=cfg, workspace_root=root)


if __name__ == "__main__":
	main()
