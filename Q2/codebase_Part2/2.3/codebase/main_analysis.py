import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models import LinearGaussianSSM
from dpf import DifferentiableParticleFilter
from soft_resampling import SoftResampling

def generate_data(T=20, true_a=0.9):
    # Generate synthetic data from the true model
    states = [tf.zeros([1, 1])]
    obs = []
    
    current_state = states[0]
    for _ in range(T):
        noise = tf.random.normal([1, 1])
        current_state = true_a * current_state + noise
        states.append(current_state)
        obs.append(current_state + tf.random.normal([1, 1]))
        
    return tf.stack(obs, axis=1) # [1, T, 1]

def run_gradient_experiment(observations, alpha_val, num_particles=100):
    """
    Computes the gradient of the loss w.r.t the transition parameter 'a'.
    """
    # Re-initialize model to random point
    model = LinearGaussianSSM(transition_a=0.5) 
    resampler = SoftResampling(alpha=alpha_val)
    dpf = DifferentiableParticleFilter(model, num_particles, resampler)

    with tf.GradientTape() as tape:
        loss = dpf(observations)
    
    grad = tape.gradient(loss, model.transition_a)
    return grad.numpy(), loss.numpy()

def main():
    print("Generating Data...")
    obs_data = generate_data(T=50, true_a=0.9)
    
    alphas = [0.0, 0.5, 0.99] # 0.0=Uniform, 0.99=Almost Hard
    results_grads = {a: [] for a in alphas}
    
    print("Running Gradient Analysis ...")
    
    # Run multiple seeds to see the VARIANCE of the gradient
    num_seeds = 50
    for seed in range(num_seeds):
        for alpha in alphas:
            grad, _ = run_gradient_experiment(obs_data, alpha)
            results_grads[alpha].append(grad)

    # Visualization
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [results_grads[a] for a in alphas]
    plt.boxplot(data_to_plot, tick_labels=[f'Alpha={a}' for a in alphas])
    
    # plt.title("Gradient Analysis in Soft Resampling")
    plt.ylabel("Gradient")
    plt.xlabel("$a$")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('gradient_bias_analysis.png')
    plt.show()

if __name__ == "__main__":
    main()