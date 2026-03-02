"""Generate stability diagnostic figures for all particle-filter models.

Outputs:
- figures/stability_flow_magnitude.png
- figures/stability_jacobian_conditioning.png
- results/stability_diagnostics_per_seed.csv
- results/stability_diagnostics_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import BASELINES, build_registry, default_config, load_or_generate_data
from utils.stability_diagnostics import (
    run_stability_diagnostics,
    run_stability_diagnostics_by_iteration,
    run_stability_diagnostics_by_iteration_scenario,
    summarize_diagnostics,
)


MODEL_ORDER = [name for name, _, _ in BASELINES]


def _save_csv(rows: list[dict], out_csv: Path, fieldnames: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_summary(summary_rows: list[dict]) -> list[dict]:
    idx = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(summary_rows, key=lambda r: idx.get(r["baseline"], 999))


def _plot_metric_line(
    ts_rows: list[dict],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    use_log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df = pd.DataFrame(ts_rows)
    for model_name in MODEL_ORDER:
        sub = df[df["baseline"] == model_name].sort_values("iteration")
        if sub.empty:
            continue
        x = sub["iteration"].to_numpy(dtype=np.int32)
        y = sub[metric_key].to_numpy(dtype=np.float64)
        ax.plot(x, y, label=model_name, linewidth=1.8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    ax.grid(axis="y", alpha=0.25)
    if use_log_scale:
        ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_line_by_scenario(
    ts_rows: list[dict],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    use_log_scale: bool = False,
) -> None:
    df = pd.DataFrame(ts_rows)
    scenarios = sorted(df["scenario"].unique().tolist())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4.2 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub_s = df[df["scenario"] == scenario]
        for model_name in MODEL_ORDER:
            sub = sub_s[sub_s["baseline"] == model_name].sort_values("iteration")
            if sub.empty:
                continue
            x = sub["iteration"].to_numpy(dtype=np.int32)
            y = sub[metric_key].to_numpy(dtype=np.float64)
            ax.plot(x, y, label=model_name, linewidth=1.6)
        ax.set_title(f"{scenario}")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if use_log_scale:
            ax.set_yscale("log")

    axes[-1].set_xlabel("Iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = ROOT
    cfg = default_config()

    rows = run_stability_diagnostics(
        cfg=cfg,
        workspace_root=root,
        build_registry_fn=build_registry,
        load_or_generate_data_fn=load_or_generate_data,
    )
    summary = summarize_diagnostics(rows)
    ts_rows = run_stability_diagnostics_by_iteration(
        cfg=cfg,
        workspace_root=root,
        build_registry_fn=build_registry,
        load_or_generate_data_fn=load_or_generate_data,
    )
    ts_rows_by_scenario = run_stability_diagnostics_by_iteration_scenario(
        cfg=cfg,
        workspace_root=root,
        build_registry_fn=build_registry,
        load_or_generate_data_fn=load_or_generate_data,
    )

    results_dir = root / "results"
    figures_dir = root / "figures"

    _save_csv(
        rows,
        out_csv=results_dir / "stability_diagnostics_per_seed.csv",
        fieldnames=[
            "scenario",
            "baseline",
            "seed",
            "flow_magnitude_mean",
            "flow_magnitude_std",
            "jacobian_conditioning_mean",
            "jacobian_conditioning_std",
        ],
    )
    _save_csv(
        summary,
        out_csv=results_dir / "stability_diagnostics_summary.csv",
        fieldnames=[
            "baseline",
            "flow_magnitude_mean",
            "flow_magnitude_std",
            "jacobian_conditioning_mean",
            "jacobian_conditioning_std",
        ],
    )
    _save_csv(
        ts_rows,
        out_csv=results_dir / "stability_diagnostics_by_iteration.csv",
        fieldnames=[
            "baseline",
            "iteration",
            "flow_magnitude_mean",
            "flow_magnitude_std",
            "jacobian_conditioning_mean",
            "jacobian_conditioning_std",
        ],
    )
    _save_csv(
        ts_rows_by_scenario,
        out_csv=results_dir / "stability_diagnostics_by_iteration_scenario.csv",
        fieldnames=[
            "scenario",
            "baseline",
            "iteration",
            "flow_magnitude_mean",
            "flow_magnitude_std",
            "jacobian_conditioning_mean",
            "jacobian_conditioning_std",
        ],
    )

    _plot_metric_line(
        ts_rows=ts_rows,
        metric_key="flow_magnitude_mean",
        title="Model Stability Diagnostics: Flow Magnitude",
        ylabel="Mean flow magnitude (L2 displacement)",
        out_path=figures_dir / "stability_flow_magnitude.png",
        use_log_scale=False,
    )
    _plot_metric_line(
        ts_rows=ts_rows,
        metric_key="jacobian_conditioning_mean",
        title="Model Stability Diagnostics: Jacobian Conditioning for Flows",
        ylabel="Mean Jacobian condition number",
        out_path=figures_dir / "stability_jacobian_conditioning.png",
        use_log_scale=True,
    )
    _plot_metric_line_by_scenario(
        ts_rows=ts_rows_by_scenario,
        metric_key="flow_magnitude_mean",
        title="Model Stability Diagnostics by Scenario: Flow Magnitude",
        ylabel="Mean flow magnitude (L2 displacement)",
        out_path=figures_dir / "stability_flow_magnitude_by_scenario.png",
        use_log_scale=False,
    )
    _plot_metric_line_by_scenario(
        ts_rows=ts_rows_by_scenario,
        metric_key="jacobian_conditioning_mean",
        title="Model Stability Diagnostics by Scenario: Jacobian Conditioning for Flows",
        ylabel="Mean Jacobian condition number",
        out_path=figures_dir / "stability_jacobian_conditioning_by_scenario.png",
        use_log_scale=True,
    )

    print(f"Saved diagnostics table: {results_dir / 'stability_diagnostics_per_seed.csv'}")
    print(f"Saved diagnostics summary: {results_dir / 'stability_diagnostics_summary.csv'}")
    print(f"Saved diagnostics by-iteration: {results_dir / 'stability_diagnostics_by_iteration.csv'}")
    print(f"Saved diagnostics by-iteration-scenario: {results_dir / 'stability_diagnostics_by_iteration_scenario.csv'}")
    print(f"Saved figure: {figures_dir / 'stability_flow_magnitude.png'}")
    print(f"Saved figure: {figures_dir / 'stability_jacobian_conditioning.png'}")
    print(f"Saved figure: {figures_dir / 'stability_flow_magnitude_by_scenario.png'}")
    print(f"Saved figure: {figures_dir / 'stability_jacobian_conditioning_by_scenario.png'}")


if __name__ == "__main__":
    main()
