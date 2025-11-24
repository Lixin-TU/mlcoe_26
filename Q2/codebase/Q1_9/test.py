import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from da_methods import EnKF, PFF_Matrix, EDH, LEDH
from run_analysis import run_experiment

# Import L96 model
sys.path.append(os.path.join(os.path.dirname(__file__), 'subroutines'))

# Define Methods
methods_to_test = ['EDH', 'LEDH', 'matrix_PFF'] 

results_db = []
# Standard settings
base_settings = {'dim': 40, 'obs_den': 2, 'dt': 0.01, 'obs_err': 0.3}

print("--- Running Optimized Comparison ---")

# ===========================================================
# Experiment A: Nonlinearity (Time Step)
# ===========================================================
print("\n[Experiment A] Nonlinearity (dt: 0.005 -> 0.030, 50 steps)")
dt_values = np.linspace(0.005, 0.030, 50) # 50 intervals

results_A = []

for dt in dt_values:
    for m in methods_to_test:
        s = base_settings.copy()
        s['dt'] = dt      
        s['method'] = m
        
        try:
            rmse, _ = run_experiment(f'Nonlin_dt_{dt:.4f}', s)
        except Exception:
            rmse = np.nan
            
        if not np.isnan(rmse) and rmse < 100: # Filter out explosions
            results_A.append({'method': m, 'dt': dt, 'rmse': rmse})
        else:
            results_A.append({'method': m, 'dt': dt, 'rmse': np.nan})

# Plot A
plt.figure(figsize=(5, 4))
for m in methods_to_test:
    data = [x for x in results_A if x['method'] == m]
    if not data: continue
    xs = [d['dt'] for d in data]
    ys = [d['rmse'] for d in data]
    # Use plot instead of scatter for smoother lines with 50 points
    plt.plot(xs, ys, '-', linewidth=2, label=m, alpha=0.8)

plt.xlabel('Nonlinearity')
plt.ylabel('RMSE')
# plt.title('Nonlinearity Robustness')
plt.legend()
plt.savefig('./Plots/Result_A_Nonlinearity.png')
plt.close()


# ===========================================================
# Experiment B: Observation Sparsity (Safe Mode)
# ===========================================================
print("\n[Experiment B] Observation Sparsity (Safe Mode)")
densities = [2, 4, 8, 10] 
results_B = []

for den in densities:
    print(f"  Testing density = 1/{den}...")
    for m in methods_to_test:
        s = base_settings.copy()
        s['obs_den'] = den
        s['method'] = m
        
        s['dt'] = 0.005  # Safe DT to isolate sparsity effect from nonlinearity
        s['inflation_fac'] = 1.05 # Add inflation to prevent filter divergence
        # =================================
        
        try:
            rmse, _ = run_experiment(f'Sparsity_{den}', s)
        except Exception as e:
            print(f"    {m} failed: {e}")
            rmse = np.nan
            
        if not np.isnan(rmse) and rmse < 100:
            results_B.append({'method': m, 'obs_den': den, 'rmse': rmse})
        else:
            results_B.append({'method': m, 'obs_den': den, 'rmse': np.nan})

# Plot B
plt.figure(figsize=(5, 4))
for m in methods_to_test:
    data = [x for x in results_B if x['method'] == m]
    if not data: continue
    xs = [d['obs_den'] for d in data]
    ys = [d['rmse'] for d in data]
    plt.plot(xs, ys, marker='s', linewidth=2, label=m)

plt.xlabel('Observation Sparsity')
plt.ylabel('RMSE')
# plt.title('Sparsity Robustness (dt=0.005, inflated)')
plt.legend()
plt.savefig('./Plots/Result_B_Sparsity.png')
plt.close()


# ===========================================================
# Experiment C: Conditioning
# ===========================================================
print("\n[Experiment C] Conditioning (Stiffness Optimization)")
errors = [0.01, 0.1, 0.3, 1.0] 

for err in errors:
    print(f"  Testing R_std = {err}...")
    for m in methods_to_test:
        s = base_settings.copy()
        s['obs_err'] = err
        s['method'] = m
        
        # If error is very small (stiff), reduce PFF step size to prevent explosion
        if err < 0.1:
            s['eps'] = 1e-3  # Small step for stiff problems
            s['max_pseudo_step'] = 500 # More steps needed if step size is small
        else:
            s['eps'] = 5e-2  # Default
        # ============================
        
        try:
            rmse, diags = run_experiment(f'Cond_err_{err}', s)
            flow = np.mean(diags.get('flow', [0]))
            print(f"    Method: {m} | RMSE: {rmse:.4f} | Avg Flow Mag: {flow:.4f}")
        except Exception:
            print(f"    Method: {m} | FAILED (Diverged)")

# ===========================================================
# Experiment D: Dimension
# ===========================================================
print("\n[Experiment D] Dimension (dim: 20 -> 100)")
dim_values = [20, 40, 60, 80, 100]
results_D = []

for dim in dim_values:
    print(f"  Testing dim = {dim}...")
    for m in methods_to_test:
        s = base_settings.copy()
        s['dim'] = dim
        s['method'] = m
        
        try:
            rmse, _ = run_experiment(f'Dim_{dim}', s)
        except Exception:
            rmse = np.nan
            
        if not np.isnan(rmse) and rmse < 100:
            results_D.append({'method': m, 'dim': dim, 'rmse': rmse})
        else:
            results_D.append({'method': m, 'dim': dim, 'rmse': np.nan})
# Plot D
plt.figure(figsize=(5, 4))  
for m in methods_to_test:
    data = [x for x in results_D if x['method'] == m]
    if not data: continue
    xs = [d['dim'] for d in data]
    ys = [d['rmse'] for d in data]
    plt.plot(xs, ys, marker='o', linewidth=2, label=m)

plt.xlabel('Dimension')
plt.ylabel('RMSE')
# plt.title('Dimension Robustness')
plt.legend()
plt.savefig('./Plots/Result_D_Dimension.png')
plt.close()

print("\nAll experiments finished. Plots saved.")