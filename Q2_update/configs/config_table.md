# Reproducibility Config Table

| Section | Key | Value |
|---|---|---|
| Dataset | `dataset_seed` | `123` |
| Dataset | `time_steps` | `100` |
| Dataset | `state_dim` | `10` |
| Dataset | `init_var` | `5.0` |
| Scenario 1 | `id` | `sigmaV2_10_sigmaW2_10` |
| Scenario 1 | `process_var` | `10.0` |
| Scenario 1 | `obs_var` | `10.0` |
| Scenario 1 | `dataset_path` | `dataset/benchmark_v10_w10.npz` |
| Scenario 2 | `id` | `sigmaV2_10_sigmaW2_1` |
| Scenario 2 | `process_var` | `10.0` |
| Scenario 2 | `obs_var` | `1.0` |
| Scenario 2 | `dataset_path` | `dataset/benchmark_v10_w1.npz` |
| Global | `global_np_seed` | `2026` |
| Global | `global_tf_seed` | `2026` |
| Evaluation | `coverage_alpha` | `0.05` |
| Shared | `num_particles` | `200` |
| DFPHS | `dfphs_ess_ratio` | `0.5` |
| DPFS | `dpfs_alpha` | `0.7` |
| DPFOT | `dpfot_epsilon` | `0.1` |
| DPFOT | `dpfot_sinkhorn_iters` | `50` |
| DPFOT-HMC | `hmc_steps` | `3` |
| DPFOT-HMC | `hmc_leapfrog_steps` | `3` |
| DPFOT-HMC | `hmc_step_size` | `0.02` |
| DPFNet-PMMH | `pmmh_steps` | `3` |
| DPFNet-PMMH | `pmmh_proposal_std` | `0.02` |
| DPFNet-PMMH | `pmmh_inner_samples` | `4` |
| DPFNet-PMMH | `pmmh_likelihood_jitter` | `0.1` |
| Flow filters | `flow_steps` | `10` |
| Flow filters | `flow_step_size` | `0.08` |
| SPFSM | `spfsm_diffusion_scale` | `1.0` |
| DPF-GradNet | `gradnet_hidden` | `64` |
