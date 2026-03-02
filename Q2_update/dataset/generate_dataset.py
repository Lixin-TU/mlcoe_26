"""Generate benchmark nonlinear SSM dataset with fixed seed and saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def transition_drift(x_prev: np.ndarray, time_index: int) -> np.ndarray:
    """Compute deterministic transition drift for the nonlinear benchmark model."""
    return 0.5 * x_prev + 25.0 * x_prev / (1.0 + x_prev**2) + 8.0 * np.cos(1.2 * time_index)


def observation_mean(x_curr: np.ndarray) -> np.ndarray:
    """Compute observation mean for the nonlinear benchmark model."""
    return (x_curr**2) / 20.0


def transition_drift_tf(x_prev: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of deterministic transition drift."""
    t = tf.cast(time_index, x_prev.dtype)
    return 0.5 * x_prev + 25.0 * x_prev / (1.0 + tf.square(x_prev)) + 8.0 * tf.cos(1.2 * t)


def observation_mean_tf(x_curr: tf.Tensor) -> tf.Tensor:
    """TensorFlow version of observation mean."""
    return tf.square(x_curr) / 20.0


def generate_dataset(
    seed: int = 123,
    time_steps: int = 100,
    state_dim: int = 10,
    process_var: float = 10.0,
    obs_var: float = 1.0,
    init_var: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic trajectory for the benchmark nonlinear SSM.

    Args:
        seed: Random seed for reproducibility.
        time_steps: Number of time points.
        state_dim: State/observation dimension.
        process_var: Process noise variance.
        obs_var: Observation noise variance.
        init_var: Initial state variance.

    Returns:
        Tuple (states, observations), both with shape (T, D).
    """
    rng = np.random.default_rng(seed)
    process_std = np.sqrt(process_var)
    obs_std = np.sqrt(obs_var)
    init_std = np.sqrt(init_var)

    x = np.zeros((time_steps, state_dim), dtype=np.float32)
    y = np.zeros((time_steps, state_dim), dtype=np.float32)
    x[0] = rng.normal(0.0, init_std, size=state_dim)
    y[0] = observation_mean(x[0]) + rng.normal(0.0, obs_std, size=state_dim)

    for t in range(1, time_steps):
        time_idx = t + 1
        drift = transition_drift(x[t - 1], time_idx)
        x[t] = drift + rng.normal(0.0, process_std, size=state_dim)
        y[t] = observation_mean(x[t]) + rng.normal(0.0, obs_std, size=state_dim)

    return x, y


def save_preview_figures(states: np.ndarray, observations: np.ndarray, out_dir: Path) -> None:
    """Save trajectory and state-observation scatter previews."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for idx in range(min(10, states.shape[1])):
        plt.plot(states[:, idx], label=f"Dim {idx + 1}", alpha=0.75)
    plt.title("State trajectories")
    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "state_trajectories.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(states[:, 0], observations[:, 0], c=np.arange(states.shape[0]), cmap="viridis", s=18, alpha=0.8)
    plt.colorbar(label="Time index")
    plt.title("Observation mapping (dimension 1)")
    plt.xlabel("State x[:, 0]")
    plt.ylabel("Observation y[:, 0]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "state_observation_scatter.png", dpi=160)
    plt.close()


def main() -> None:
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate benchmark nonlinear SSM dataset.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--process-var", type=float, default=10.0)
    parser.add_argument("--obs-var", type=float, default=1.0)
    parser.add_argument("--init-var", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="dataset/benchmark_dataset.npz")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states, observations = generate_dataset(
        seed=args.seed,
        time_steps=args.time_steps,
        state_dim=args.state_dim,
        process_var=args.process_var,
        obs_var=args.obs_var,
        init_var=args.init_var,
    )

    np.savez(
        output_path,
        states=states,
        observations=observations,
        seed=np.int32(args.seed),
        time_steps=np.int32(args.time_steps),
        state_dim=np.int32(args.state_dim),
        process_var=np.float32(args.process_var),
        obs_var=np.float32(args.obs_var),
        init_var=np.float32(args.init_var),
    )

    if not args.skip_plots:
        save_preview_figures(states=states, observations=observations, out_dir=output_path.parent)

    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()