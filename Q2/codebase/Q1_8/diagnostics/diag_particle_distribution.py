import os
import numpy as np
import matplotlib.pyplot as plt

def plot_particle_distribution():
    # Load results
    try:
        results_path = os.path.join(os.path.dirname(__file__), '../PFF_results.npz')
        data = np.load(results_path)
    except FileNotFoundError:
        print("Error: PFF_results.npz not found. Please run PFF.py first.")
        return

    prior = data['prior']
    X = data['X'] # Posterior (Analysis)
    da_intv = int(data['da_intv'])
    
    # Settings for the plot
    # We want to look at the FIRST assimilation step (t = 20 if da_intv=20)
    # obs_time_index = 0 corresponds to first analysis
    obs_idx = 0 
    time_step_actual = (obs_idx + 1) * da_intv # e.g. 20
    
    # Indices for variables (Figure 3 in paper uses x19 and x20)
    # Python is 0-based, so x19 -> index 18, x20 -> index 19
    dim_unobs = 18 
    dim_obs = 19
    
    # Extract particles
    # Prior: stored in 'prior' variable [dim, np, total_obs]
    parts_prior = prior[[dim_unobs, dim_obs], :, obs_idx]
    
    # Posterior: stored in 'X' variable [dim, np, nt]
    # Note: X index corresponds to time steps. 
    # The analysis result at t=20 is stored at index 20 (since t=0 is IC)
    parts_post = X[[dim_unobs, dim_obs], :, time_step_actual]
    
    # Plotting
    plt.figure(figsize=(7, 4))
    
    # Plot Prior (Black circles)
    plt.scatter(parts_prior[0, :], parts_prior[1, :], 
                s=40, facecolors='none', edgecolors='k', linewidth=1.5, label='Prior')
    
    # Plot Posterior (Red circles, filled)
    plt.scatter(parts_post[0, :], parts_post[1, :], 
                s=40, c='r', edgecolors='r', label='Posterior')
    
    plt.title(f'Matrix-valued Kernel')
    plt.xlabel(f'x{dim_unobs+1} (unobserved)')
    plt.ylabel(f'x{dim_obs+1} (observed)')
    # plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.axis('equal') # Keep aspect ratio
    plt.xlim([-2, 12])
    plt.ylim([-2, 10])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_particle_distribution()