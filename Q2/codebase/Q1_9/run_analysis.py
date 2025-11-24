import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from da_methods import PFF_Matrix, EDH, LEDH
from tqdm import tqdm # Install with pip install tqdm for progress bar, or remove

# Import L96 model
sys.path.append(os.path.join(os.path.dirname(__file__), 'subroutines'))
from L96_RK4 import L96_RK4
from H_linear import H_linear

def run_experiment(exp_name, settings):
    """
    Executes a single experiment configuration.
    settings: dict containing dim, obs_den, method_name, etc.
    """
    # 1. Setup Parameters
    dim = settings.get('dim', 40)
    F = settings.get('F', 8.0)
    dt = settings.get('dt', 0.01) # Controls Nonlinearity
    obs_den = settings.get('obs_den', 2) # Controls Sparsity
    obs_err = settings.get('obs_err', 0.3) # Controls Conditioning
    nt = settings.get('nt', 100)
    da_intv = settings.get('da_intv', 20)
    method_name = settings.get('method', 'EnKF')
    
    np_particles = 30
    
    # 2. Initialize Method
    # Map user friendly names to classes
    if method_name == 'matrix_PFF' or method_name == 'PFF_Matrix':
        da_solver = PFF_Matrix()
    elif method_name == 'EDH':
        da_solver = EDH()
    elif method_name == 'LEDH':
        da_solver = LEDH()
    else:
        raise ValueError(f"Unknown Method: {method_name}")
        
    # 3. Generate Truth and Obs
    warm_nt = 500
    Xt = np.zeros((dim, warm_nt + nt))
    Xt[:, 0] = F
    Xt[int(dim/2), 0] += 0.1 # Perturbation
    
    for t in range(warm_nt + nt - 1):
        Xt[:, t+1] = L96_RK4(Xt[:, t:t+1], dt, F).flatten()
        
    # Generate Observations
    obs_input = np.arange(obs_den-1, dim, obs_den)
    ny_obs = len(obs_input)
    inner_domain = [[idx] for idx in obs_input]
    R = (obs_err**2) * np.eye(ny_obs)
    
    obs_data = {} # store obs by time index
    for t in range(nt - 1):
        if (t+1) % da_intv == 0:
            true_state = Xt[:, warm_nt + t + 1]
            # H_linear assumes input is (dim_inner, 1)
            # We construct obs vector
            curr_obs = np.zeros(ny_obs)
            for i in range(ny_obs):
                val = true_state[inner_domain[i]]
                curr_obs[i] = H_linear(val.reshape(-1,1)) + np.random.normal(0, obs_err)
            obs_data[t+1] = curr_obs

    # 4. Run DA Cycle
    X = np.zeros((dim, np_particles, nt))
    X[:, :, 0] = Xt[:, warm_nt].reshape(-1, 1) + np.random.normal(0, 1, (dim, np_particles))
    
    rmse_history = []
    diag_history = {'flow': [], 'cond': []}
    
    try: 
        for t in range(nt - 1):
            # Forecast
            X[:, :, t+1] = L96_RK4(X[:, :, t], dt, F)
            
            # Inf/NaN 
            if not np.all(np.isfinite(X[:, :, t+1])):
                raise ValueError("Model Diverged (Infinity/NaN detected)")

            # Analysis
            if (t+1) in obs_data:
                current_obs = obs_data[t+1]
                params = {
                    'inner_domain': inner_domain, 
                    'r_influ': 4,
                    'inflation_fac': 1.1,
                    'obs_den': obs_den,
                    'max_pseudo_step': 100,
                    'alpha': 1.0/np_particles,
                    'io_local': 1
                }
                
                # CALL DA INTERFACE
                X_ana, diags = da_solver.update(X[:, :, t+1], current_obs, R, params)
                X[:, :, t+1] = X_ana
                
                # Record Diagnostics
                if 'flow_magnitude' in diags: diag_history['flow'].append(diags['flow_magnitude'])
                if 'cond_num' in diags: diag_history['cond'].append(diags['cond_num'])
                if 'cond_B' in diags: diag_history['cond'].append(diags['cond_B'])
                
            # Calc RMSE
            truth = Xt[:, warm_nt + t + 1]
            mean_state = np.mean(X[:, :, t+1], axis=1)
            rmse = np.sqrt(np.mean((mean_state - truth)**2))
            rmse_history.append(rmse)
            
    except (ValueError, np.linalg.LinAlgError) as e: 
        print(f"!!! Experiment Crashed at t={t}: {e}")
        return np.nan, diag_history 

    avg_rmse = np.mean(rmse_history[da_intv:]) # Ignore spinup
    print(f"[{exp_name}] Method: {method_name} | RMSE: {avg_rmse:.4f}")
    
    return avg_rmse, diag_history