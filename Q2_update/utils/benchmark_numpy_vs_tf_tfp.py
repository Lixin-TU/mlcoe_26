"""Benchmark NumPy vs TF/TFP for representative HMC sampling workload.

Outputs:
- results/numpy_vs_tf_tfp_benchmark.csv
- results/numpy_vs_tf_tfp_benchmark_summary.md
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def _load_class(file_path: Path, class_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(file_path.stem.replace("-", "_"), str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _obs_log_prob_numpy(x: np.ndarray, observation: np.ndarray, obs_var: float) -> np.ndarray:
    loc = (x * x) / 20.0
    c = np.log(2.0 * np.pi * obs_var)
    return np.sum(-0.5 * (c + ((observation[None, :] - loc) ** 2) / obs_var), axis=-1).astype(np.float32)


@tf.function
def _tf_tfp_sampling_loop(
    x0: tf.Tensor,
    observation: tf.Tensor,
    repeats: tf.Tensor,
    obs_var: tf.Tensor,
    step_size: tf.Tensor,
    leapfrog_steps: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    def target_log_prob_fn(x: tf.Tensor) -> tf.Tensor:
        loc = tf.square(x) / 20.0
        c = tf.math.log(2.0 * tf.constant(np.pi, dtype=x.dtype) * obs_var)
        return tf.reduce_sum(-0.5 * (c + tf.square(observation[tf.newaxis, :] - loc) / obs_var), axis=-1)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps,
    )

    i0 = tf.constant(0, dtype=tf.int32)
    seed0 = tf.constant([12345, 67890], dtype=tf.int32)
    state0 = x0
    kr0 = kernel.bootstrap_results(state0)

    def cond(i: tf.Tensor, _state: tf.Tensor, _kr: Any, _seed: tf.Tensor) -> tf.Tensor:
        del _state, _kr, _seed
        return i < repeats

    def body(i: tf.Tensor, state: tf.Tensor, kr: Any, seed: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, Any, tf.Tensor]:
        seeds = tf.random.experimental.stateless_split(seed, num=2)
        next_seed = seeds[0]
        step_seed = seeds[1]
        state, kr = kernel.one_step(state, kr, seed=step_seed)
        return i + 1, state, kr, next_seed

    _, state_f, kr_f, _ = tf.while_loop(
        cond,
        body,
        loop_vars=(i0, state0, kr0, seed0),
        parallel_iterations=1,
    )
    accept_rate = tf.reduce_mean(tf.cast(kr_f.is_accepted, tf.float32))
    return state_f, accept_rate


def _benchmark_numpy(
    repeats: int,
    num_particles: int,
    state_dim: int,
    obs_var: float,
    hmc_step_size: float,
    hmc_leapfrog_steps: int,
) -> tuple[float, float]:
    root = Path(__file__).resolve().parents[1]
    cls = _load_class(root / "DPFNet-HMC-numpy" / "DPFNet-HMC-numpy.py", "DPFNet_HMC_Numpy")
    model = cls(
        state_dim=state_dim,
        num_particles=num_particles,
        process_var=10.0,
        obs_var=obs_var,
        init_var=5.0,
        hmc_steps=1,
        hmc_leapfrog_steps=hmc_leapfrog_steps,
        hmc_step_size=hmc_step_size,
    )

    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=(num_particles, state_dim)).astype(np.float32)
    observation = rng.normal(0.0, 1.0, size=(state_dim,)).astype(np.float32)

    t0 = time.perf_counter()
    accept_acc = 0.0
    for _ in range(int(repeats)):
        x_prop = model._hmc_transition(x, observation, rng)
        lp_cur = _obs_log_prob_numpy(x, observation, obs_var)
        lp_new = _obs_log_prob_numpy(x_prop, observation, obs_var)
        accept_acc += float(np.mean((lp_new >= lp_cur).astype(np.float32)))
        x = x_prop
    dt = time.perf_counter() - t0
    accept_rate = accept_acc / max(1, int(repeats))
    return float(dt), float(accept_rate)


def _benchmark_tf_tfp(
    repeats: int,
    num_particles: int,
    state_dim: int,
    obs_var: float,
    hmc_step_size: float,
    hmc_leapfrog_steps: int,
) -> tuple[float, float]:
    tf.random.set_seed(2026)
    x0 = tf.random.normal((num_particles, state_dim), dtype=tf.float32)
    observation = tf.random.normal((state_dim,), dtype=tf.float32)

    _ = _tf_tfp_sampling_loop(
        x0=x0,
        observation=observation,
        repeats=tf.constant(2, dtype=tf.int32),
        obs_var=tf.constant(obs_var, dtype=tf.float32),
        step_size=tf.constant(hmc_step_size, dtype=tf.float32),
        leapfrog_steps=tf.constant(hmc_leapfrog_steps, dtype=tf.int32),
    )

    t0 = time.perf_counter()
    _, accept_rate = _tf_tfp_sampling_loop(
        x0=x0,
        observation=observation,
        repeats=tf.constant(int(repeats), dtype=tf.int32),
        obs_var=tf.constant(obs_var, dtype=tf.float32),
        step_size=tf.constant(hmc_step_size, dtype=tf.float32),
        leapfrog_steps=tf.constant(hmc_leapfrog_steps, dtype=tf.int32),
    )
    dt = time.perf_counter() - t0
    return float(dt), float(accept_rate.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs TF/TFP for representative HMC sampling workload.")
    parser.add_argument("--repeats", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--num-particles", type=int, default=200)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--obs-var", type=float, default=10.0)
    parser.add_argument("--hmc-step-size", type=float, default=0.02)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results_path = root / "results" / "numpy_vs_tf_tfp_benchmark.csv"
    summary_path = root / "results" / "numpy_vs_tf_tfp_benchmark_summary.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for repeats in [int(r) for r in args.repeats]:
        np_time, np_accept = _benchmark_numpy(
            repeats=repeats,
            num_particles=int(args.num_particles),
            state_dim=int(args.state_dim),
            obs_var=float(args.obs_var),
            hmc_step_size=float(args.hmc_step_size),
            hmc_leapfrog_steps=int(args.hmc_leapfrog_steps),
        )
        tf_time, tf_accept = _benchmark_tf_tfp(
            repeats=repeats,
            num_particles=int(args.num_particles),
            state_dim=int(args.state_dim),
            obs_var=float(args.obs_var),
            hmc_step_size=float(args.hmc_step_size),
            hmc_leapfrog_steps=int(args.hmc_leapfrog_steps),
        )

        rows.append(
            {
                "repeats": repeats,
                "impl": "NumPy",
                "runtime_sec": np_time,
                "samples_per_sec": repeats / max(np_time, 1e-12),
                "accept_proxy": np_accept,
            }
        )
        rows.append(
            {
                "repeats": repeats,
                "impl": "TF/TFP",
                "runtime_sec": tf_time,
                "samples_per_sec": repeats / max(tf_time, 1e-12),
                "accept_proxy": tf_accept,
            }
        )

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["repeats", "impl", "runtime_sec", "samples_per_sec", "accept_proxy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        grouped.setdefault(int(row["repeats"]), {})[str(row["impl"])] = float(row["runtime_sec"])

    lines = [
        "# NumPy vs TF/TFP Benchmark (Representative HMC Sampling)",
        "",
    ]
    for repeats in sorted(grouped.keys()):
        np_t = grouped[repeats].get("NumPy", np.nan)
        tf_t = grouped[repeats].get("TF/TFP", np.nan)
        speedup = np_t / max(tf_t, 1e-12)
        lines.append(
            f"- repeats={repeats}: NumPy={np_t:.4f}s, TF/TFP={tf_t:.4f}s, speedup(NumPy/TF)={speedup:.3f}x"
        )


    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved benchmark CSV: {results_path}")
    print(f"Saved benchmark summary: {summary_path}")


if __name__ == "__main__":
    main()
