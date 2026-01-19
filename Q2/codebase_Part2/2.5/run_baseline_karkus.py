
import os
import sys
import abc

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
    from filterflow.base import Module
    from filterflow.observation import LinearObservationModel
    from filterflow.proposal import BootstrapProposalModel
    from filterflow.resampling.base import ResamplerBase
    from filterflow.transition import RandomWalkModel
    from filterflow.resampling import NeffCriterion
except ImportError as e:
    print(f"Error importing filterflow: {e}")
    sys.exit(1)

# --- Implement Karkus (18) Soft Resampler ---

class KarkusSoftResampler(ResamplerBase):
    """
    Implements the Soft Resampling from Karkus et al. (2018).
    'Particle Filter Networks with Likelihood Gradients'
    
    It mixes the particle weights with a uniform distribution (controlled by alpha)
    to define the resampling probability q.
    Then it corrects the weights of the resampled particles using importance sampling:
    w_{new} = w_{old} / q
    """
    DIFFERENTIABLE = True # It claims to allow some gradients through weights

    def __init__(self, alpha: float, name='KarkusSoftResampler'):
        super(KarkusSoftResampler, self).__init__(name=name)
        assert 0.0 <= alpha <= 1.0
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """
        :param state: State object containing particles and log_weights
        :param flags: boolean tensor (batch_size,) indicating whether to resample
        :return: new State
        """
        # We only implement the logic. The SMC handling of flags (conditional execution) involves 
        # tf.cond usually inside the library, but here apply receives flags.
        # FilterFlow's SMC usually handles the 'if' logic by passing flags to the resampler 
        # or wrapping it. Let's look at how we should behave.
        # In filterflow/resampling/base.py, there is a 'resample' function that mixes 
        # resampled and non-resampled based on flags.
        # We will calculate the resampled state for ALL, and then select based on flags.
        
        batch_size = state.batch_size
        n_particles = state.n_particles
        
        log_weights = state.log_weights # (Batch, N)
        particles = state.particles # (Batch, N, Dim)

        # 1. Compute Proposal Distribution q
        # Uniform log prob = -log(N)
        log_uniform = -tf.math.log(tf.cast(n_particles, tf.float32))
        
        # log_q_unnorm = logsumexp( [log(alpha) + log_w, log(1-alpha) + log_uniform] )
        # We need to handle alpha=1 or alpha=0 cautiously to avoid log(0). 
        # However, for fixed alpha=0.5 it's fine.
        
        # Safe log_alpha terms
        term1 = log_weights + tf.math.log(self.alpha + 1e-8)
        term2 = log_uniform + tf.math.log(1.0 - self.alpha + 1e-8)
        
        # Broadcast term2 to match term1 shape (Batch, N)
        term2 = tf.broadcast_to(term2, tf.shape(term1))

        log_q_unnorm = tf.reduce_logsumexp(tf.stack([term1, term2], axis=-1), axis=-1)
        
        # Normalize q
        q_normalizer = tf.reduce_logsumexp(log_q_unnorm, axis=1, keepdims=True)
        log_q = log_q_unnorm - q_normalizer # Normalized proposal log prob
        
        # 2. Resample Indices from q
        # Use stateless categorical sampling compatible with FilterFlow's Tensor seeds
        if seed is None:
            indices = tf.random.categorical(log_q, n_particles, dtype=tf.int32) 
        else:
            indices = tf.random.stateless_categorical(log_q, n_particles, seed=seed, dtype=tf.int32)
        
        # indices shape: (Batch, N)
        
        # 3. Permute Particles and Weights
        # We need to gather validation.
        # Gather requires batch indices helper
        batch_indices = tf.expand_dims(tf.range(batch_size), axis=1) # (B, 1)
        batch_indices = tf.tile(batch_indices, [1, n_particles]) # (B, N)
        
        gather_indices = tf.stack([batch_indices, indices], axis=-1) # (B, N, 2)
        
        new_particles = tf.gather_nd(particles, gather_indices)
        
        # Gather old weights (to compute the correction)
        # Note: Karkus uses w_old / q_old. 
        # The gathered weight IS w_old[indices].
        gathered_log_weights = tf.gather_nd(log_weights, gather_indices)
        gathered_log_q = tf.gather_nd(log_q, gather_indices)
        
        # 4. Weight Correction (Importance Sampling)
        # new_w = old_w / q
        # new_log_w = gathered_log_w - gathered_log_q
        
        # Explanation: We sampled index 'i' with prob q[i].
        # To maintain the distribution estimate, we weight it by w[i]/q[i].
        resampled_log_weights = gathered_log_weights - gathered_log_q
        
        # Re-normalize just in case (though technically should be normalized in expectation)
        final_log_weights = resampled_log_weights - tf.reduce_logsumexp(resampled_log_weights, axis=1, keepdims=True)

        # 5. Apply Flags (Select between resampled and original)
        # flags shape (Batch,) -> reshape to (Batch, 1) or (Batch, 1, 1) for broadcasting
        # We use filterflow's helper if available, or manual tf.where
        
        flags_broadcast_w = tf.reshape(flags, [batch_size, 1])
        flags_broadcast_p = tf.reshape(flags, [batch_size, 1, 1])
        
        final_particles = tf.where(flags_broadcast_p, new_particles, particles)
        final_weights = tf.where(flags_broadcast_w, final_log_weights, log_weights)
        
        return State(final_particles, final_weights)

# --- Main Script ---

def main():
    print("Setting up the Karkus (18) Baseline (Soft Resampling)...")
    
    tfd = tfp.distributions

    # Data Generation
    seed = 42
    rng = np.random.RandomState(seed)
    T = 150
    noise = 0.5

    linspace = np.linspace(0., 5., T)
    sine = np.sin(linspace)
    noisy_sine = sine + rng.normal(0., noise, T)
    observations_dataset = tf.data.Dataset.from_tensor_slices(noisy_sine.astype(np.float32))

    # Model Parameters
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

    # Karkus (18) Soft Resampler
    # alpha = 0.5 (Mixing uniform and particle weights equally for proposal)
    alpha_value = 0.5
    resampling_method = KarkusSoftResampler(alpha=alpha_value)

    print(f"Initializing SMC with Karkus Soft Resampler (alpha={alpha_value})...")

    smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampling_method)

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

    # --- Metrics ---
    # 1. Accuracy
    estimated_path = mean_particles[:, 0, 0, 0].numpy()
    rmse = np.sqrt(np.mean((estimated_path - sine)**2))
    print(f"Metric - Accuracy (RMSE) for Batch 0: {rmse:.4f}")
    
    # 2. Differentiability
    print("Metric - Differentiability: PARTIAL.")
    print("Karkus (18) allows gradients to flow through the WEIGHTS via the importance sampling correction (w_new = w_old / q).")
    print("However, the INDEX SELECTION is still discrete (Multinomial) and gradients are blocked there.")
    print("It is often called 'Differentiable' because it avoids the complete gradient blockage of standard resampling by maintaining a dependency chain in the weight values.")

    # Visualization
    print("Generating result plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(linspace, sine, 'k--', label='True State', linewidth=2)
    ax.plot(linspace, noisy_sine, 'r.', label='Observations', alpha=0.6)
    
    std_path = std_particles[:, 0, 0, 0].numpy()
    
    ax.plot(linspace, estimated_path, 'm-', label='Estimated Mean (Karkus Soft)', linewidth=2)
    ax.fill_between(linspace, estimated_path - 2*std_path, estimated_path + 2*std_path, color='m', alpha=0.2, label='Confidence Interval')
    
    ax.set_title(f'Karkus (18) Baseline: Soft Resampling (alpha={alpha_value})')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_plot = os.path.join(current_dir, 'karkus_baseline_result.png')
    plt.savefig(output_plot)
    print(f"Result plot saved to {output_plot}")
    print("Done.")

if __name__ == "__main__":
    main()
