
import os
import sys

# Add the Corenflos(21) directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'Corenflos(21)')
sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
    from filterflow import SMC, State, mean, std
    from filterflow.observation import LinearObservationModel
    from filterflow.proposal import BootstrapProposalModel
    from filterflow.resampling import NeffCriterion, SystematicResampler
    from filterflow.transition import RandomWalkModel
except ImportError as e:
    print(f"Error importing filterflow: {e}")
    sys.exit(1)

def main():
    print("Setting up the Wen (21) Baseline...")
    print("Code Analysis of 'Wen (21)/methods/dpf.py' reveals it uses Hard Systematic Resampling.")
    print("In the context of Inference (without semi-supervised training), this is algorithmically")
    print("equivalent to the Jonschkowski (18) DPF baseline.")
    print("Running Systematic Resampling...")

    tfd = tfp.distributions
    seed = 42
    rng = np.random.RandomState(seed)
    T = 150
    noise = 0.5

    linspace = np.linspace(0., 5., T)
    sine = np.sin(linspace)
    noisy_sine = sine + rng.normal(0., noise, T)
    observations_dataset = tf.data.Dataset.from_tensor_slices(noisy_sine.astype(np.float32))

    sigma_x = 0.5
    sigma_y = 1.
    observation_matrix = tf.eye(1)
    transition_matrix = tf.eye(1)

    transition_noise = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_x]))
    observation_error = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_y]))
    
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    observation_model = LinearObservationModel(observation_matrix, observation_error)
    proposal_model = BootstrapProposalModel(transition_model)

    resampling_criterion = NeffCriterion(0.5, is_relative=True)
    
    # Wen (21) uses Systematic Resampling in the provided code
    resampling_method = SystematicResampler()

    smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

    batch_size = 5
    n_particles = 50
    dimension = 1
    
    initial_particles = rng.normal(0., 1., [batch_size, n_particles, dimension]).astype(np.float32)
    initial_state = State(tf.convert_to_tensor(initial_particles))

    print(f"Running SMC with {n_particles} particles over {T} time steps...")

    # Run
    state_series = smc(initial_state, observations_dataset, T, return_final=False, seed=555)
    
    # --- Evaluation & Diagnostics ---
    
    # 1. RMSE Accuracy
    mean_particles = mean(state_series, keepdims=True) # (T, B, 1, D)
    estimated_path = mean_particles[:, 0, 0, 0].numpy()
    rmse = np.sqrt(np.mean((estimated_path - sine)**2))
    
    # 2. Log-Likelihood
    log_likelihoods = state_series.log_likelihoods.numpy() # (B,)
    # Handle the structure of log_likelihoods carefully
    # It seems to be a scalar or array depending on the run.
    # In batch mode (B=5), it should be (5,).
    try:
        final_ll = float(log_likelihoods[0])
    except (TypeError, IndexError):
        # Case where it might be (150, 5) if return_final=False doesn't apply to LL or something else
        # Or if it's already a scalar
        if np.ndim(log_likelihoods) == 0:
             final_ll = float(log_likelihoods)
        else:
             final_ll = float(log_likelihoods.flatten()[0])
    
    # 3. Effective Sample Size (ESS)
    # ESS = 1 / sum(w^2) = 1 / sum(exp(2*log_w)) where w is normalized
    # filterflow State has log_weights.
    # We need to extract them. state_series.probs might not be directly available, let's check State structure
    # state_series.log_weights is (T, B, N)
    log_weights_series = state_series.log_weights
    
    # Normalize weights manually to be sure
    max_log_w = tf.reduce_max(log_weights_series, axis=2, keepdims=True)
    w_unnorm = tf.exp(log_weights_series - max_log_w)
    w_norm = w_unnorm / tf.reduce_sum(w_unnorm, axis=2, keepdims=True)
    
    ess_series = 1.0 / tf.reduce_sum(tf.square(w_norm), axis=2) # (T, B)
    avg_ess = tf.reduce_mean(ess_series[:, 0]).numpy()
    
    print("-" * 30)
    print("EVALUATION REPORT (Wen 21 Baseline)")
    print("-" * 30)
    print(f"1. Accuracy (RMSE):     {rmse:.4f}")
    print(f"   (Lower is better)")
    print(f"2. Log-Likelihood:      {final_ll:.4f}")
    print(f"   (Higher is better)")
    print(f"3. Avg ESS (Batch 0):   {avg_ess:.2f} / {n_particles}")
    print(f"   (Higher indicates less degeneracy)")
    print(f"4. Resampling Type:     Hard / Systematic")
    print(f"   (Gradients blocked: Yes)")
    print("-" * 30)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Tracking
    std_particles = std(state_series, mean_particles)
    std_path = std_particles[:, 0, 0, 0].numpy()
    
    ax1.plot(linspace, sine, 'k--', label='True State', linewidth=2)
    ax1.plot(linspace, noisy_sine, 'r.', label='Observations', alpha=0.6)
    ax1.plot(linspace, estimated_path, 'c-', label='Estimated (Wen 21)', linewidth=2)
    ax1.fill_between(linspace, estimated_path - 2*std_path, estimated_path + 2*std_path, color='c', alpha=0.2, label='Confidence Interval')
    ax1.set_ylabel('State')
    ax1.set_title('Wen (21) Baseline: Systematic Resampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ESS
    ax2.plot(linspace, ess_series[:, 0].numpy(), 'k-', label='ESS (Batch 0)')
    ax2.axhline(y=n_particles/2, color='r', linestyle='--', label='Resampling Threshold (N/2)')
    ax2.set_ylabel('Effective Sample Size')
    ax2.set_xlabel('Time')
    ax2.set_title('Particle Health (ESS)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    output_plot = os.path.join(current_dir, 'wen_baseline_result.png')
    plt.savefig(output_plot)
    print(f"Result plot saved to {output_plot}")

if __name__ == "__main__":
    main()
