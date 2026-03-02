## question2_polished

TensorFlow/TFP reimplementation of particle-filter baselines.

### Project Layout

- `baselines/`: each baseline implementation.
- `utils/`: evaluation and transport utilities.
- `dataset/`: synthetic benchmark data generation.
- `tests/`: unit and integration tests.
- `configs/`: reproducibility seeds and config table.

### Implemented Baselines

- `DFPHS` (hard resampling)
- `DPFS` (soft resampling)
- `DPFOT` (entropy-regularized OT)
- `DPFOT-HMC` (OT + Hamiltonian Monte Carlo)
- `IPFPF` (deterministic particle flow)
- `SPFSM` (stochastic particle flow)
- `DPF-GradNet` (amortized transport network)
- `DPFNet-HMC` (DPF-GradNet transport + Hamiltonian Monte Carlo)
- `DPFNet-PMMH` (DPF-GradNet transport + pseudo-marginal Metropolis-Hastings)

### Reproducibility

- Seeds: `configs/seeds.csv`
- Config table: `configs/config_table.md`
- Single entry point: `run.py`

### Scenarios

`run.py` evaluates all baselines under both required settings:

- `σ_V^2 = 10`, `σ_W^2 = 10`
- `σ_V^2 = 10`, `σ_W^2 = 1`

### Quick Start

From this folder:

1. Generate dataset:
	- `python dataset/generate_dataset.py --output dataset/benchmark_dataset.npz`
2. Run all baselines and reproduce outputs:
	- `python run.py`
3. Analyze regularization-iterations-speed trade-offs for OT/Sinkhorn-family baselines:
	- `python utils/evaluate_ot_tradeoffs.py`
4. Compare `DPFNet-PMMH` vs `DPFNet-HMC` across scenarios on differentiability-bias, OT regularization, and gradient stability/variance:
	- `python utils/compare_dpfnet_pmmh_hmc.py`
5. Run a brief benchmark comparing NumPy vs TF/TFP on representative HMC sampling workload:
	- `python utils/benchmark_numpy_vs_tf_tfp.py`
6. Plot benchmark results:
	- `python utils/plot_numpy_vs_tf_benchmark.py`

### Outputs

- `results/main_results.csv`: per-seed metrics with `scenario` column (`rmse`, `rmse_percent`, `coverage`, `coverage_percent`, `ESS`,`runtime`, `peak memory`).
- `results/main_summary.csv`: aggregated mean/std table (includes percentage metrics).
- `results/used_config.json`: full run config used.
- `figures/main_figure.png`: scenario-wise comparison figure.
- `figures/ess_over_time_sigmaV2_10_sigmaW2_10.png`: ESS bar chart every 25 timesteps with error bars.
- `figures/ess_over_time_sigmaV2_10_sigmaW2_1.png`: ESS bar chart every 25 timesteps with error bars.
	- Model order: `SPFSM/IPFPF/DPFS/DPFOT/DPFOT-HMC/DPF-GradNet/DPFNet-HMC (proposed)`.
- `results/ot_tradeoff_per_seed.csv`: OT trade-off per-seed results over regularization × iterations grids.
- `results/ot_tradeoff_summary.csv`: aggregated OT trade-off summary table.
- `figures/ot_tradeoff_<model>_<scenario>.png`: heatmaps for RMSE, ESS, and runtime trade-offs.
- `results/pmmh_hmc_differentiability_bias_*.csv`: per-seed and summary tables for differentiability-bias trade-off.
- `results/pmmh_hmc_ot_regularization_*.csv`: per-seed and summary tables for OT regularization effects.
- `results/pmmh_hmc_gradient_stability_*.csv`: per-seed and summary tables for gradient stability and variance.
- `figures/pmmh_hmc_comparison_<scenario>.png`: one-page scenario comparison (three panels: differentiability-bias, OT regularization effects, gradient stability/variance).
- `results/numpy_vs_tf_tfp_benchmark.csv`: benchmark runtime/throughput table over workload sizes.
- `results/numpy_vs_tf_tfp_benchmark_summary.md`: concise interpretation and TF/TFP rationale for large evaluation counts.
- `figures/numpy_vs_tf_tfp_benchmark.png`: runtime and speedup plot for NumPy vs TF/TFP.

