# Differentiable Particle Filters Comparison

This project implements and compares four different Differentiable Particle Filter (DPF) algorithms using the `filterflow` framework.

## Project Structure

This folder contains baselines from the following papers:
1.  **Corenflos et al. (2021)**: Differentiable Particle Filtering via Entropy-Regularized Optimal Transport.
2.  **Jonschkowski et al. (2018)**: Differentiable Particle Filters (Hard/Systematic Resampling baseline).
3.  **Karkus et al. (2018)**: Particle Filter Networks (Soft Resampling).
4.  **Wen et al. (2021)**: End-to-end semi-supervised learning for DPFs (Uses Systematic Resampling for inference).



### 1. Simple Baseline Comparison (1D Gaussian Random Walk)
Run the following scripts to evaluate each algorithm on a simple 1D tracking task (Linear Gaussian Model).

- **Corenflos (21)**:
  ```bash
  python run_baseline.py
  ```
- **Jonschkowski (18)**:
  ```bash
  python run_baseline_jonschkowski.py
  ```
- **Karkus (18)**:
  ```bash
  python run_baseline_karkus.py
  ```
- **Wen (21)**:
  ```bash
  python run_baseline_wen.py
  ```

### 2. Complex Scenario Comparison (3D Lorenz Attractor)
Run the complex scenario script to compare all algorithms on a high-dimensional non-linear system (Lorenz 63) with Non-Gaussian noise (Student-t process noise + Laplace observation noise).

```bash
python run_complex.py
```
*(Note: If the file is named `run_complex_scenario.py`, use that instead).*
