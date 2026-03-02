"""Plot NumPy vs TF/TFP benchmark results.

Input:
- results/numpy_vs_tf_tfp_benchmark.csv

Output:
- figures/numpy_vs_tf_tfp_benchmark.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NumPy vs TF/TFP benchmark results.")
    parser.add_argument("--input", type=str, default="results/numpy_vs_tf_tfp_benchmark.csv")
    parser.add_argument("--output", type=str, default="figures/numpy_vs_tf_tfp_benchmark.png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_csv = (root / args.input).resolve()
    output_png = (root / args.output).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Benchmark CSV is empty.")

    reps = sorted(df["repeats"].astype(int).unique().tolist())
    impls = ["NumPy", "TF/TFP"]

    runtime = {impl: [] for impl in impls}
    throughput = {impl: [] for impl in impls}
    for r in reps:
        sub = df[df["repeats"].astype(int) == r]
        for impl in impls:
            row = sub[sub["impl"] == impl]
            if row.empty:
                runtime[impl].append(np.nan)
                throughput[impl].append(np.nan)
            else:
                runtime[impl].append(float(row.iloc[0]["runtime_sec"]))
                throughput[impl].append(float(row.iloc[0]["samples_per_sec"]))

    speedup = np.array(runtime["NumPy"], dtype=np.float64) / np.maximum(np.array(runtime["TF/TFP"], dtype=np.float64), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    ax0.plot(reps, runtime["NumPy"], marker="o", label="NumPy")
    ax0.plot(reps, runtime["TF/TFP"], marker="o", label="TF/TFP")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("Sampling repetitions (log scale)")
    ax0.set_ylabel("Runtime [s] (log scale)")
    ax0.set_title("Runtime comparison")
    ax0.grid(alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(reps, speedup, marker="s", color="tab:green")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sampling repetitions (log scale)")
    ax1.set_ylabel("Speedup = NumPy runtime / TF runtime")
    ax1.set_title("Relative speedup (higher is better for TF/TFP)")
    ax1.grid(alpha=0.3)

    fig.suptitle("NumPy vs TF/TFP Benchmark for HMC Sampling", y=1.02)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved benchmark plot: {output_png}")


if __name__ == "__main__":
    main()
