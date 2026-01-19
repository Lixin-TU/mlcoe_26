import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
from models import LinearGaussianSSM
from dpf import DifferentiableParticleFilter
from ot_resampling import RegularisedTransform

def generate_data(T=20, true_a=0.9):
    """Generates structured data consistent with Task A"""
    states = [tf.zeros([1, 1])]
    obs = []
    current_state = states[0]
    for _ in range(T):
        noise = tf.random.normal([1, 1])
        current_state = true_a * current_state + noise
        states.append(current_state)
        # Observation is state + noise
        obs.append(current_state + tf.random.normal([1, 1]))
    return tf.stack(obs, axis=1) # [1, T, 1]

def main():
    
    # 1. Use STRUCTURED data (Targeting true_a=0.9)
    # This aligns the gradient direction with Task A
    obs_data = generate_data(T=20, true_a=0.9)
    
    # Tuning Grid
    epsilons = [0.1, 0.5, 1.0, 2.5] # Added intermediate points
    iterations_list = [10, 30, 50, 100]
    
    results_bias_var = {} 
    results_time = {}     
    
    # --- Part 1: Bias-Variance Trade-off (Fixed Iterations=50, Vary Epsilon) ---
    print("Analyzing Bias/Variance...")
    fixed_iter = 50
    for eps in epsilons:
        grads = []
        for _ in range(30): # 30 seeds for statistics
            model = LinearGaussianSSM(transition_a=0.5) # Start at 0.5, target is 0.9
            resampler = RegularisedTransform(epsilon=eps, max_iter=fixed_iter)
            dpf = DifferentiableParticleFilter(model, 50, resampler)
            
            with tf.GradientTape() as tape:
                loss = dpf(obs_data)
            
            g = tape.gradient(loss, model.transition_a)
            if g is not None and not np.isnan(g):
                grads.append(g.numpy())
        
        if len(grads) > 0:
            results_bias_var[eps] = (np.mean(grads), np.var(grads))
            print(f"Eps={eps}: Mean Grad={np.mean(grads):.2f} (Expected approx negative)")
        else:
            print(f"Eps={eps}: All NaNs (Unstable)")

    # --- Part 2: Speed Analysis (Fixed Epsilon=0.5, Vary Iterations) ---
    print("Analyzing Speed...")
    fixed_eps = 0.5
    for it in iterations_list:
        times = []
        # Warmup
        model = LinearGaussianSSM(transition_a=0.5)
        resampler = RegularisedTransform(epsilon=fixed_eps, max_iter=it)
        dpf = DifferentiableParticleFilter(model, 50, resampler)
        _ = dpf(obs_data)

        for _ in range(20):
            t0 = time.time()
            _ = dpf(obs_data) 
            times.append(time.time() - t0)
        results_time[it] = np.mean(times)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bias-Variance
    # We define "Bias" roughly as distance from the Task A 'True' gradient (approx -32)
    eps_vals = sorted(results_bias_var.keys())
    means = [results_bias_var[e][0] for e in eps_vals]
    stds = [np.sqrt(results_bias_var[e][1]) for e in eps_vals]
    
    ax1.errorbar(eps_vals, means, yerr=stds, fmt='-o', capsize=5, label='OT Gradient')
    # Add a reference line from soft computing (Approx -30)
    ax1.axhline(-30, color='r', linestyle='--', alpha=0.5, label='Approx. Target Gradient')
    
    ax1.set_title(f"Gradient Estimate vs Epsilon (Iter={fixed_iter})")
    ax1.set_xlabel("Epsilon (Regularization)")
    ax1.set_ylabel("Gradient Estimate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speed
    it_vals = sorted(results_time.keys())
    t_vals = [results_time[i] for i in it_vals]
    ax2.plot(it_vals, t_vals, marker='s', color='green')
    ax2.set_title(f"Execution Time vs Sinkhorn Iterations")
    ax2.set_xlabel("Sinkhorn Iterations")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()

if __name__ == "__main__":
    main()