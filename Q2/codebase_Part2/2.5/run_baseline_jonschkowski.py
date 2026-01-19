
import os
import sys

# Add the Corenflos(21) directory to sys.path so we can import filterflow
# This assumes this script is located in the parent directory of Corenflos(21)
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'Corenflos(21)')
sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Import filterflow modules
try:
    from filterflow import SMC, State, mean, std
    from filterflow.observation import LinearObservationModel
    from filterflow.proposal import BootstrapProposalModel
    from filterflow.resampling import NeffCriterion, SystematicResampler
    from filterflow.transition import RandomWalkModel
except ImportError as e:
    print(f"Error importing filterflow: {e}")
    print(f"Please make sure the 'Corenflos(21)' directory is in {current_dir}")
    sys.exit(1)

def main():
    print("Setting up the Jonschkowski (18) baseline (Standard Systematic Resampling)...")
    print("Note: Jonschkowski et al. (2018) propose a Differentiable Particle Filter framework.")
    print("In the absence of a learned proposer, their method (when using hard resampling) is equivalent ")
    print("to a Standard Particle Filter with gradients blocked through resampling.")
    print("This script runs that configuration as a baseline for comparison.")

    tfd = tfp.distributions

    # Generate artificial data (Same as Corenflos baseline for fair comparison)
    seed = 42
    rng = np.random.RandomState(seed)
    T = 150
    noise = 0.5

    linspace = np.linspace(0., 5., T)
    sine = np.sin(linspace)
    noisy_sine = sine + rng.normal(0., noise, T)
    
    observations_dataset = tf.data.Dataset.from_tensor_slices(noisy_sine.astype(np.float32))

    # Set the model parameters
    sigma_x = 0.5
    sigma_y = 1.
    observation_matrix = tf.eye(1)
    transition_matrix = tf.eye(1)

    transition_noise = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_x]))
    observation_error = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_y]))
    
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    observation_model = LinearObservationModel(observation_matrix, observation_error)
    proposal_model = BootstrapProposalModel(transition_model)

    # Resampling Criterion
    resampling_criterion = NeffCriterion(0.5, is_relative=True)

    # Jonschkowski (18) Baseline: Use Systematic Resampling (Standard, non-differentiable resampling)
    # The 'RegularisedTransform' used in Corenflos (21) is the Differentiable DetResampling.
    # Here we use the standard approach.
    resampling_method = SystematicResampler()

    print("Initializing SMC with Systematic Resampler...")

    # The SMC object
    smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

    # The Initial state
    batch_size = 5
    n_particles = 50
    dimension = 1
    
    initial_particles = rng.normal(0., 1., [batch_size, n_particles, dimension]).astype(np.float32)
    initial_state = State(tf.convert_to_tensor(initial_particles))

    print(f"Running SMC with {n_particles} particles over {T} time steps...")

    # Run
    state_series = smc(initial_state, observations_dataset, T, return_final=False, seed=555)

    log_likelihoods = state_series.log_likelihoods
    mean_particles = mean(state_series, keepdims=True)
    std_particles = std(state_series, mean_particles)

    print("Inference complete.")
    print(f"Top 5 final mean states (batch 0): {mean_particles[-1, 0, 0].numpy()}")
    print(f"Final log likelihood (batch 0): {log_likelihoods[0].numpy()}")

    # --- Metrics Consideration ---
    # 1. Accuracy (RMSE)
    # We compare the estimated mean path against the true 'sine' state.
    # Note: 'sine' is numpy (T,), mean_particles is (T, B, 1, D)
    
    estimated_path = mean_particles[:, 0, 0, 0].numpy() # Take first batch
    rmse = np.sqrt(np.mean((estimated_path - sine)**2))
    print(f"Metric - Accuracy (RMSE) for Batch 0: {rmse:.4f}")

    # 2. Efficiency (Effective Sample Size - Proxy)
    # We can't easily measure runtime here without benchmarking, but Systematic Resampling 
    # is generally O(N) or O(N log N) very efficient.
    # FilterFlow calculates ESS internally to trigger resampling.
    
    # 3. Differentiability
    print("Metric - Differentiability: NO. Gradients cannot flow through the integers/indices sampled in Systematic Resampling.")

    # Visualization
    print("Generating result plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(linspace, sine, 'k--', label='True State', linewidth=2)
    ax.plot(linspace, noisy_sine, 'r.', label='Observations', alpha=0.6)
    
    std_path = std_particles[:, 0, 0, 0].numpy()
    
    ax.plot(linspace, estimated_path, 'g-', label='Estimated Mean (Jonschkowski Baseline)', linewidth=2)
    ax.fill_between(linspace, estimated_path - 2*std_path, estimated_path + 2*std_path, color='g', alpha=0.2, label='Confidence Interval')
    
    ax.set_title('Jonschkowski (18) Baseline: Standard Systematic Resampling')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_plot = os.path.join(current_dir, 'jonschkowski_baseline_result.png')
    plt.savefig(output_plot)
    print(f"Result plot saved to {output_plot}")
    print("Done.")

if __name__ == "__main__":
    main()
