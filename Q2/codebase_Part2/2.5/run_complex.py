
import os
import sys
import abc

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, 'Corenflos(21)')
sys.path.append(module_path)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
    from filterflow import SMC, State, mean, std
    from filterflow.base import Module
    from filterflow.observation.base import ObservationModelBase
    from filterflow.transition.base import TransitionModelBase
    from filterflow.proposal import BootstrapProposalModel
    from filterflow.resampling import NeffCriterion, SystematicResampler, RegularisedTransform
    from filterflow.resampling.base import ResamplerBase
except ImportError as e:
    print(f"Error importing filterflow: {e}")
    sys.exit(1)

tfd = tfp.distributions

# ==========================================
# 1. Define Complex Models (Lorenz 63 + Non-Gaussian)
# ==========================================

class Lorenz63TransitionModel(TransitionModelBase):
    """
    3D Non-linear Chaotic System: Lorenz 63 Attractor
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    def __init__(self, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0, 
                 noise_scale=0.1, df=3.0, name='Lorenz63Transition'):
        super(Lorenz63TransitionModel, self).__init__(name=name)
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
        # Heavy-tailed Student-t Distribution for Process Noise (Non-Gaussian)
        self.noise_dist = tfd.StudentT(df=df, loc=0.0, scale=noise_scale)

    def sample(self, state: State, inputs: tf.Tensor, seed=None):
        # state.particles shape: [Batch, N, 3]
        x, y, z = tf.unstack(state.particles, axis=-1)
        
        # Euler integration for dynamics
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        x_new = x + dx * self.dt
        y_new = y + dy * self.dt
        z_new = z + dz * self.dt
        
        deterministic_next = tf.stack([x_new, y_new, z_new], axis=-1)
        
        # Add non-Gaussian noise
        noise = self.noise_dist.sample([state.batch_size, state.n_particles, 3], seed=seed)
        
        return deterministic_next + noise

    def loglikelihood(self, prior_state: State, proposed_state: State, inputs: tf.Tensor):
        # For Bootstrap proposal, this is not strictly needed for weights, 
        # but required by interface.
        # Calculating log_prob of StudentT is supported.
        # Note: recovering the noise realization to check prob
        
        # Re-compute deterministic step to find the noise residual
        x, y, z = tf.unstack(prior_state.particles, axis=-1)
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        pred_x = x + dx * self.dt
        pred_y = y + dy * self.dt
        pred_z = z + dz * self.dt
        deterministic_next = tf.stack([pred_x, pred_y, pred_z], axis=-1)
        
        residual = proposed_state.particles - deterministic_next
        return tf.reduce_sum(self.noise_dist.log_prob(residual), axis=-1)


class NonLinearObservationModel(ObservationModelBase):
    """
    Observations: 
    1. x coordinate directly
    2. Distance from origin (nonlinear)
    
    Noise: Laplace Distribution (Peakier than Gaussian)
    """
    def __init__(self, noise_scale=0.5, name='NonLinearObservation'):
        super(NonLinearObservationModel, self).__init__(name=name)
        # Laplace noise
        self.noise_dist = tfd.Laplace(loc=0.0, scale=noise_scale)
        self.noise_scale = noise_scale

    def loglikelihood(self, state: State, observation: tf.Tensor):
        # state.particles: [Batch, N, 3]
        # observation: [Batch, 2] (x, distance) provided by dataset iteration
        
        x, y, z = tf.unstack(state.particles, axis=-1)
        
        # Predicted measurements
        # 1. Observe x
        obs_1 = x 
        # 2. Observe distance to origin sqrt(x^2 + y^2 + z^2)
        obs_2 = tf.sqrt(x**2 + y**2 + z**2 + 1e-6)
        
        predicted_obs = tf.stack([obs_1, obs_2], axis=-1)
        
        # Robust Broadcasting
        # Observation shape can be [ObsDim] (from unbatched dataset) or [Batch, ObsDim]
        # We want [Batch, N, ObsDim] (to match predicted_obs) or [1, 1, ObsDim] for broadcasting
        
        # If shape is [2], make it [1, 1, 2]
        # If shape is [Batch, 2], make it [Batch, 1, 2]
        
        if len(observation.shape) == 1:
             observation_broadcast = tf.reshape(observation, [1, 1, -1])
        elif len(observation.shape) == 2:
             observation_broadcast = tf.expand_dims(observation, axis=1)
        else:
             observation_broadcast = observation # Hope it works
        
        # Calculate Log Prob under Laplace
        log_prob = self.noise_dist.log_prob(observation_broadcast - predicted_obs)
        
        # Sum over observation dimensions
        return tf.reduce_sum(log_prob, axis=-1)

# Re-implement Karkus Soft Resampler here to be self-contained
class KarkusSoftResampler(ResamplerBase):
    DIFFERENTIABLE = True
    def __init__(self, alpha: float, name='KarkusSoft'):
        super(KarkusSoftResampler, self).__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def apply(self, state: State, flags: tf.Tensor, seed=None):
        batch_size = state.batch_size
        n_particles = state.n_particles
        log_weights = state.log_weights
        particles = state.particles

        log_uniform = -tf.math.log(tf.cast(n_particles, tf.float32))
        term1 = log_weights + tf.math.log(self.alpha + 1e-8)
        term2 = log_uniform + tf.math.log(1.0 - self.alpha + 1e-8)
        term2 = tf.broadcast_to(term2, tf.shape(term1))
        
        log_q_unnorm = tf.reduce_logsumexp(tf.stack([term1, term2], axis=-1), axis=-1)
        q_normalizer = tf.reduce_logsumexp(log_q_unnorm, axis=1, keepdims=True)
        log_q = log_q_unnorm - q_normalizer
        
        if seed is None:
            indices = tf.random.categorical(log_q, n_particles, dtype=tf.int32)
        else:
            indices = tf.random.stateless_categorical(log_q, n_particles, seed=seed, dtype=tf.int32)
            
        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, n_particles])
        gather_indices = tf.stack([batch_indices, indices], axis=-1)
        
        new_particles = tf.gather_nd(particles, gather_indices)
        gathered_log_weights = tf.gather_nd(log_weights, gather_indices)
        gathered_log_q = tf.gather_nd(log_q, gather_indices)
        
        resampled_log_weights = gathered_log_weights - gathered_log_q
        final_log_weights = resampled_log_weights - tf.reduce_logsumexp(resampled_log_weights, axis=1, keepdims=True)

        flags_broadcast_p = tf.reshape(flags, [batch_size, 1, 1])
        flags_broadcast_w = tf.reshape(flags, [batch_size, 1])
        
        return State(
            tf.where(flags_broadcast_p, new_particles, particles),
            tf.where(flags_broadcast_w, final_log_weights, log_weights)
        )

# ==========================================
# 2. Main Compare Script
# ==========================================

def generate_lorenz_data(T, dt, process_noise_scale, obs_noise_scale, seed=42):
    np.random.seed(seed)
    
    # Lorenz Params
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    # Initial state near attractor
    state = np.array([1.0, 1.0, 1.0])
    
    true_states = []
    observations = []
    
    for _ in range(T):
        # Dynamics
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        state = state + np.array([dx, dy, dz]) * dt
        
        # Add Student-t Noise (Approximated by standard non-gaussian sampling in numpy)
        # Using df=3 for heavy tails
        proc_noise = np.random.standard_t(df=3, size=3) * process_noise_scale
        state = state + proc_noise
        
        true_states.append(state)
        
        # Observation
        # 1. x
        # 2. distance
        ox = state[0]
        od = np.sqrt(np.sum(state**2))
        
        # Add Laplace Noise
        obs_noise = np.random.laplace(loc=0.0, scale=obs_noise_scale, size=2)
        obs = np.array([ox, od]) + obs_noise
        
        observations.append(obs)
        
    return np.array(true_states, dtype=np.float32), np.array(observations, dtype=np.float32)

def main():
    print("================================================================")
    print("COMPLEX SCENARIO: Lorenz 63 (3D) + Student-t Noise + Laplace Obs")
    print("================================================================")
    
    T = 200 # Longer horizon
    dt = 0.02
    n_particles = 100 # Need more particles for 3D
    batch_size = 1 # Single batch for clear analysis
    
    # Generate Ground Truth Data
    true_states, observations = generate_lorenz_data(T, dt, process_noise_scale=0.2, obs_noise_scale=1.0)
    
    # Create TF Dataset
    observations_dataset = tf.data.Dataset.from_tensor_slices(observations)

    # Models
    transition_model = Lorenz63TransitionModel(dt=dt, noise_scale=0.2)
    observation_model = NonLinearObservationModel(noise_scale=1.0)
    proposal_model = BootstrapProposalModel(transition_model)
    resampling_criterion = NeffCriterion(0.5, is_relative=True)

    # Initial State
    initial_particles = np.random.normal(0, 1, [batch_size, n_particles, 3]).astype(np.float32) + np.array([1,1,1], dtype=np.float32)
    initial_state_tf = State(tf.convert_to_tensor(initial_particles))

    # --- Run Comparisons ---
    results = {}
    
    methods = {
        "Jonschkowski (Hard)": SystematicResampler(),
        "Wen (Systematic)": SystematicResampler(),
        "Karkus (Soft)": KarkusSoftResampler(alpha=0.5),
        "Corenflos (Regularized)": RegularisedTransform(epsilon=tf.constant(0.1)) # Lower epsilon for higher precision
    }
    
    for name, resampler in methods.items():
        print(f"Running {name}...")
        smc = SMC(observation_model, transition_model, proposal_model, resampling_criterion, resampler)
        
        # Run
        state_series = smc(initial_state_tf, observations_dataset, T, return_final=False, seed=tf.constant([1, 1]))
        
        # Calculate Metrics
        mean_particles = mean(state_series, keepdims=False).numpy() # [T, B, 3] -> [T, 1, 3]
        if mean_particles.ndim == 3: mean_particles = mean_particles[:, 0, :]
        
        # RMSE (Average over 3 dimensions)
        diff = mean_particles - true_states
        rmse = np.sqrt(np.mean(diff**2))
        
        # Store
        results[name] = {
            "rmse": rmse,
            "path": mean_particles
        }
        print(f"  -> RMSE: {rmse:.4f}")

    # --- Visualization ---
    print("Generating 3D plot...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot True
    ax.plot(true_states[:,0], true_states[:,1], true_states[:,2], 'k-', label='True Lorenz Attractor', alpha=0.5, linewidth=1)
    
    # Plot Methods
    colors = ['r', 'g', 'm', 'b']
    for i, (name, res) in enumerate(results.items()):
        path = res["path"]
        ax.plot(path[:,0], path[:,1], path[:,2], label=f"{name} (RMSE={res['rmse']:.2f})", color=colors[i], linewidth=2)
        
    # ax.set_title("3D Particle Filter Comparison on Lorenz 63 (Student-t/Laplace Noise)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.savefig(os.path.join(current_dir, 'complex_scenario_3d_result.png'))
    print(f"Comparison plot saved to 'complex_scenario_3d_result.png'")
    
    # 2D Error Plot (Time Series)
    fig2, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Error X', 'Error Y', 'Error Z']
    time_axis = np.arange(T) * dt
    
    for dim in range(3):
        for i, (name, res) in enumerate(results.items()):
            path = res["path"]
            # Calculate Absolute Error for this dimension
            error_dim = np.abs(path[:, dim] - true_states[:, dim])
            axes[dim].plot(time_axis, error_dim, label=name, color=colors[i], alpha=0.8)
            
        axes[dim].set_ylabel(labels[dim])
        axes[dim].grid(True, alpha=0.3)
        # axes[dim].set_yscale('log') # Optional: Log scale helps see differences
    
    axes[0].legend()
    # axes[0].set_title("Absolute State Estimation Error over Time (Lorenz 63)")
    plt.savefig(os.path.join(current_dir, 'complex_scenario_error_series.png'))
    print(f"Error plot saved to 'complex_scenario_error_series.png'")
    print("Done.")

if __name__ == "__main__":
    main()
