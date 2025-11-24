import numpy as np
import matplotlib.pyplot as plt
import os

def plot_scalar_distribution():
    # Construct path to PFF_scalar_results.npz (in parent directory)
    results_path = os.path.join(os.path.dirname(__file__), '../PFF_scalar_results.npz')
    
    try:
        data = np.load(results_path)
    except FileNotFoundError:
        print(f"Error: {results_path} not found.")
        print("Please run 'python PFF_scalar.py' in the root directory first.")
        return

    prior = data['prior']
    X = data['X'] 
    da_intv = int(data['da_intv'])
    
    # Plotting for t=20 (First DA Step)
    obs_idx = 0 
    time_step_actual = (obs_idx + 1) * da_intv 
    
    dim_unobs = 18 # x19
    dim_obs = 19   # x20
    
    parts_prior = prior[[dim_unobs, dim_obs], :, obs_idx]
    parts_post = X[[dim_unobs, dim_obs], :, time_step_actual]
    
    plt.figure(figsize=(7, 4))
    
    # Plot Prior (Black)
    plt.scatter(parts_prior[0, :], parts_prior[1, :], 
                s=40, facecolors='none', edgecolors='k', linewidth=1.5, label='Prior')
    
    # Plot Posterior (Red) - Scalar Kernel Result
    plt.scatter(parts_post[0, :], parts_post[1, :], 
                s=40, c='r', edgecolors='r', label='Posterior (Scalar)')
    
    plt.title(f'Scalar Kernel')
    plt.xlabel(f'x{dim_unobs+1} (unobserved)')
    plt.ylabel(f'x{dim_obs+1} (observed)')
    # plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.axis('equal') 
    plt.xlim([-2, 12])
    plt.ylim([-2, 10])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_scalar_distribution()