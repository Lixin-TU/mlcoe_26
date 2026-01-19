
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
    from filterflow.resampling import NeffCriterion, RegularisedTransform
    from filterflow.transition import RandomWalkModel
except ImportError as e:
    print(f"Error importing filterflow: {e}")
    print(f"Please make sure the 'Corenflos(21)' directory is in {current_dir}")
    sys.exit(1)

def main():
    print("Setting up the particle filter baseline...")
    
    tfd = tfp.distributions

    # Generate artificial data.
    # We use a fixed seed for reproducibility
    seed = 42
    rng = np.random.RandomState(seed)
    T = 150 # Number of time steps - Duration of the simulation
    noise = 0.5 # Noise level for observations

    # Here we simply use a noisy sine function as the ground truth process
    linspace = np.linspace(0., 5., T)
    sine = np.sin(linspace)
    # Generate observations by adding noise to the sine wave
    noisy_sine = sine + rng.normal(0., noise, T)
    
    # Create a TensorFlow dataset from the observations
    # filterflow expects the dataset to yield observations step by step
    observations_dataset = tf.data.Dataset.from_tensor_slices(noisy_sine.astype(np.float32))

    # Set the model parameters
    sigma_x = 0.5 # Standard deviation for transition noise
    sigma_y = 1.  # Standard deviation for observation noise
    observation_matrix = tf.eye(1) # Identity observation matrix
    transition_matrix = tf.eye(1)  # Identity transition matrix

    # Define the transition noise distribution (Process noise)
    # We explicitly set loc=0 and scale_diag=sigma_x for a zero-mean random walk
    transition_noise = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_x]))
    
    # Define the observation noise distribution (Measurement noise)
    observation_error = tfd.MultivariateNormalDiag(loc=tf.zeros([1]), scale_diag=tf.constant([sigma_y]))
    
    # Instantiate the models components
    transition_model = RandomWalkModel(transition_matrix, transition_noise)
    observation_model = LinearObservationModel(observation_matrix, observation_error)
    proposal_model = BootstrapProposalModel(transition_model)

    # Let's resample when the Effective Sample Size (ESS) drops below 50%
    resampling_criterion = NeffCriterion(0.5, is_relative=True)

    # And use Differentiable Ensemble Transform (DET) resampling
    # epsilon controls the regularization strength for the optimal transport
    epsilon = tf.constant(0.5)
    resampling_method = RegularisedTransform(epsilon)

    # Indication of starting the SMC setup
    print("Initializing SMC...")

    # The SMC object combines all components
    smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

    # The Initial state setup
    batch_size = 5 # Number of independent chains
    n_particles = 50 # Number of particles per chain
    dimension = 1 # State dimension
    
    # Sample initial particles from a standard normal distribution
    initial_particles = rng.normal(0., 1., [batch_size, n_particles, dimension]).astype(np.float32)
    initial_state = State(tf.convert_to_tensor(initial_particles))

    print(f"Running SMC with {n_particles} particles over {T} time steps...")

    # Run the particle filter
    # return_final=False ensures we get the state at every time step
    state_series = smc(initial_state, observations_dataset, T, return_final=False, seed=555)

    # Extract estimates from the results
    log_likelihoods = state_series.log_likelihoods
    mean_particles = mean(state_series, keepdims=True)
    std_particles = std(state_series, mean_particles) # the mean argument is optional

    print("Inference complete.")
    print(f"Top 5 final mean states (batch 0): {mean_particles[-1, 0, 0].numpy()}")
    print(f"Final log likelihood (batch 0): {log_likelihoods[0].numpy()}")

    # --- Metrics Consideration ---
    # 1. Accuracy (RMSE)
    estimated_path = mean_particles[:, 0, 0, 0].numpy()
    rmse = np.sqrt(np.mean((estimated_path - sine)**2))
    print(f"Metric - Accuracy (RMSE) for Batch 0: {rmse:.4f}")

    # Visualization
    print("Generating result plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(linspace, sine, 'k--', label='True State (Sine)', linewidth=2)
    ax.plot(linspace, noisy_sine, 'r.', label='Observations', alpha=0.6)
    
    # Plot the estimated mean path for the first batch index
    # mean_particles has shape [T, batch_size, 1, dimension] (because keepdims=True)
    # We want a 1D array for plotting [T]
    estimated_path = mean_particles[:, 0, 0, 0].numpy()
    std_path = std_particles[:, 0, 0, 0].numpy()
    
    print(f"Shape of estimated_path for plotting: {estimated_path.shape}")
    
    ax.plot(linspace, estimated_path, 'b-', label='Estimated Mean', linewidth=2)
    # Plot confidence interval (2 standard deviations)
    ax.fill_between(linspace, estimated_path - 2*std_path, estimated_path + 2*std_path, color='b', alpha=0.2, label='Confidence Interval (2 std)')
    
    ax.set_title('Particle Filter Baseline (FilterFlow)')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_plot = os.path.join(current_dir, 'baseline_result.png')
    plt.savefig(output_plot)
    print(f"Result plot saved to {output_plot}")
    print("Done.")

if __name__ == "__main__":
    main()
