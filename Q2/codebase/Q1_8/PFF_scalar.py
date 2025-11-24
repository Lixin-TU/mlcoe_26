import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Add subroutines to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'subroutines'))

from L96_RK4 import L96_RK4
from H_linear import H_linear
from H_linear_adjoint import H_linear_adjoint
from H_linear_sum import H_linear_sum
from H_linear_sum_adjoint import H_linear_sum_adjoint
from inv_SVD import inv_SVD
from adjoint_pseudoinverse import adjoint_pseudoinverse

# ==========================================
# PFF with SCALAR Kernel
# Based on PFF.py, modified to use Scalar Kernel
# ==========================================

# %% Namelist
DA_run   = 1                  
noDA_run = 1                  
nt       = 200                
t_start  = 0                  
warm_nt  = 1000               
gen_ens  = 1                  
np_particles = 30             

# parameters for the L96 model:
dim = 40                      
F   = 8.0                     
dt  = 0.01                    

# settings for generating the ensemble
Q     = 2 * np.eye(dim)       
Q_inv = np.linalg.inv(Q)

# settings for DA/PFF 
alpha    = 1.0/np_particles   
max_pseudo_step     = 150     
eps_init            = 5e-2    
stop_cri            = 1e-3    
stop_cri_percentage = 0.05    
min_learning_rate   = 1e-5    

io_local       = 1            
r_influ        = 4            
io_gauss_prior = 1            
inflation_fac  = 1.25         
tune_C  = 5.0/inflation_fac   
pre_cond  = 1                 
cond_num  = -5                

# settings for observation
da_intv  = 20                 
obs_den      = 2
obs_input    = np.arange(obs_den-1, dim, obs_den) 
ny_obs = len(obs_input)
inner_domain = [[idx] for idx in obs_input]

# generate the observation error
np.random.seed(0) 
obs_err   = 0.3
R         = obs_err**2 * np.eye(ny_obs)
total_obs = int(nt/da_intv)
obs_rnd   = np.random.multivariate_normal(np.zeros(ny_obs), R, total_obs).T

if gen_ens == 1:
    prior = np.zeros((dim, np_particles, total_obs))
obs = np.zeros((ny_obs, total_obs))
norm_grad_KL = np.zeros((total_obs, max_pseudo_step))

# %% Integrate the L96 model (warm up)
Xt = np.zeros((dim, warm_nt + nt))
Xt[:, 0] = F * np.ones(dim)
Xt[7::8, 0] = F + 1 

for t in range(warm_nt + nt - 1):
    Xt[:, t+1] = L96_RK4(Xt[:, t].reshape(-1, 1), dt, F).flatten()

# %% Generate the ensemble
if gen_ens == 1:
    ctlmean = Xt[:, warm_nt] + np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 1).flatten()
    X = np.zeros((dim, np_particles, nt))
    X[:, :, 0] = np.random.multivariate_normal(ctlmean, Q, np_particles).T

# %% Integration & DA
if noDA_run == 1:
    XnoDA = np.zeros((dim, np_particles, nt))
    XnoDA[:, :, 0] = X[:, :, 0]
    for t in range(nt - 1):
        XnoDA[:, :, t+1] = L96_RK4(XnoDA[:, :, t], dt, F)

if DA_run == 1:
    t = t_start
    while t < nt - 1:
        print(f'start timestep t={t+1} (Scalar Kernel Run)')
        
        if (t+1) % da_intv == 0:
            io_obs = 1
            io_pff = 1
            print(f'start DA at t={t+1}')
        else:
            io_obs = 0
            io_pff = 0
            
        X[:, :, t+1] = L96_RK4(X[:, :, t], dt, F)
        
        if io_obs == 1:
            obs_time_idx = int((t+1)/da_intv) - 1 
            for i in range(ny_obs):
                inner_ind = inner_domain[i]
                val = Xt[inner_ind, warm_nt + t + 1]
                obs[i, obs_time_idx] = H_linear(val.reshape(-1,1)) + obs_rnd[i, obs_time_idx]

        # Step 3 -- Particle Flow Filter (PFF)
        if io_pff == 1:
            X_tmp = X[:, :, t+1]
            X_mean = np.tile(np.mean(X_tmp, axis=1, keepdims=True), (1, np_particles))
            
            diff = X_tmp - X_mean
            C = inflation_fac * (diff @ diff.T) / (np_particles - 1) / np_particles
            
            if io_local == 1:
                tmp = np.zeros((dim, dim))
                for i in range(1, 3 * r_influ + 1):
                    d1 = np.diag(np.ones(dim - i), i)
                    d2 = np.diag(np.ones(dim - i), -i)
                    d3 = np.diag(np.ones(i), -(dim - i))
                    d4 = np.diag(np.ones(i), dim - i)
                    tmp = tmp + np.exp(-i**2 / r_influ**2) * (d1 + d2 + d3 + d4)
                mask = tmp + np.eye(dim)
                C = C * mask
                C_inv = inv_SVD(C, cond_num, 1)
            else:
                C_inv = inv_SVD(C, cond_num, 1)
                
            B = C * np_particles
            B_inv = C_inv / np_particles
            
            if pre_cond == 1:
                qn = B / inflation_fac
            else: 
                # fallback if needed, though pre_cond=1 is recommended
                qn = np.eye(dim) 

            s = 0 
            ct = 0
            norm_grad_KL[obs_time_idx, 0] = 1e8
            eps = eps_init * np.ones(max_pseudo_step)
            
            pseudo_X = np.zeros((dim, np_particles, max_pseudo_step))
            pseudo_X[:, :, 0] = X_mean + (X_tmp - X_mean)
            grad_KL = np.zeros((dim, np_particles, max_pseudo_step))
            
            # Use B_inv for scalar kernel calculation (Mahalanobis distance)
            # Scaling by alpha as per original formula structure: K ~ exp(-0.5 * dx' * (alpha*B)^-1 * dx)
            # So matrix in middle is B_inv / alpha
            Kernel_Prec = B_inv / alpha

            while s < max_pseudo_step:
                if s > 0 and norm_grad_KL[obs_time_idx, max(s-1, 0)] <= stop_cri:
                    break
                    
                Hx_ens = np.zeros((ny_obs, np_particles))
                dHdx = np.zeros((ny_obs, dim, np_particles))
                
                for i in range(ny_obs):
                    inner_ind = inner_domain[i]
                    current_X_sub = pseudo_X[inner_ind, :, s].reshape(len(inner_ind), -1)
                    Hx_ens[i, :] = H_linear(current_X_sub).flatten()
                    tmp_dHdx = H_linear_adjoint(current_X_sub)
                    for part_idx in range(np_particles):
                        for k, d_idx in enumerate(inner_ind):
                             dHdx[i, d_idx, part_idx] = tmp_dHdx[k, part_idx]
                
                tmp_grad_log_post = np.zeros((dim, np_particles))
                p_obs = np.zeros((dim, np_particles))
                p_bkg = np.zeros((dim, np_particles))
                
                for i in range(np_particles):
                    if io_gauss_prior == 1:
                        p_bkg[:, i] = -B_inv @ (pseudo_X[:, i, s] - np.mean(X_tmp, axis=1))
                    HT = dHdx[:, :, i].T
                    p_obs[:, i] = HT @ (np.linalg.inv(R) @ (obs[:, obs_time_idx] - Hx_ens[:, i]))
                    tmp_grad_log_post[:, i] = p_obs[:, i] + p_bkg[:, i]
                    
                # ========================================================
                # Modified Particle Flow Kernel (SCALAR Version)
                # ========================================================
                grad_KL[:, :, s] = 0 # Reset gradient
                
                # Pairwise calculations
                for i in range(np_particles):
                    for j in range(np_particles):
                        # Difference vector
                        diff = pseudo_X[:, i, s] - pseudo_X[:, j, s]
                        
                        # Scalar Kernel Calculation (Mahalanobis distance)
                        # dist_sq = diff.T * (B_inv / alpha) * diff
                        dist_sq = diff @ Kernel_Prec @ diff
                        
                        # Scalar Kernel value
                        kij = np.exp(-0.5 * dist_sq)
                        
                        # Gradient of Kernel with respect to x_i
                        # grad_K_ij = -kij * Kernel_Prec * diff
                        grad_kij = -kij * (Kernel_Prec @ diff)
                        
                        # Update KL gradient
                        # Note: In scalar case, K is scalar, multiplies grad_log_post (vector)
                        term1 = kij * tmp_grad_log_post[:, j]
                        term2 = grad_kij 
                        
                        # Check symmetry logic from original code:
                        # Original used: (tmp_K * grad_log_post + tmp_grad_K) / np
                        # Here we sum over j directly
                        
                        grad_KL[:, i, s] += (term1 + term2) / np_particles

                # ========================================================
                
                norm_val = np.sqrt(np.sum(grad_KL[:, :, s]**2) / (dim * np_particles))
                norm_grad_KL[obs_time_idx, s] = norm_val
                
                print(f"iteration s={s+1} norm ={norm_val/norm_grad_KL[obs_time_idx, 0]*100:.2f} eps = {eps[s]}")
                
                # Adaptive Learning Rate (Same as before)
                next_step = False
                if s == 0:
                     stop_cri = stop_cri_percentage * norm_grad_KL[obs_time_idx, 0]
                     pseudo_X[:, :, s+1] = pseudo_X[:, :, s] + eps[s] * (qn @ grad_KL[:, :, s])
                     s += 1
                     ct += 1
                     next_step = True
                     
                if not next_step:
                    if eps[s] < min_learning_rate:
                        print('[Note] learning rate too small, break!')
                        break
                    prev_max = 0
                    if s > 0: prev_max = np.max(norm_grad_KL[obs_time_idx, s-1:s])
                    
                    if s >= 1 and norm_grad_KL[obs_time_idx, s] > 1.02 * prev_max:
                        eps[s-1:] = eps[s] / 1.5
                        s = s - 1
                        ct = 0
                        print(f'[Note] The eps has been changed to {eps[s]} ... Redo')
                    elif ct >= 7 and norm_grad_KL[obs_time_idx, s] <= 1.02 * prev_max:
                        eps[s:] = eps[s] * 1.5
                        if s + 1 < max_pseudo_step:
                            pseudo_X[:, :, s+1] = pseudo_X[:, :, s] + eps[s] * (qn @ grad_KL[:, :, s])
                        s += 1
                        ct = 0
                    else:
                        if s + 1 < max_pseudo_step:
                            pseudo_X[:, :, s+1] = pseudo_X[:, :, s] + eps[s] * (qn @ grad_KL[:, :, s])
                        s += 1
                        ct += 1
                
                if s >= max_pseudo_step - 1:
                    break
                    
            s_end = s
            prior[:, :, obs_time_idx] = X[:, :, t+1] 
            X[:, :, t+1] = pseudo_X[:, :, s_end] 
            
        t += 1

# Save as Scalar results
np.savez('PFF_scalar_results.npz', 
         X=X, Xt=Xt, XnoDA=XnoDA, prior=prior, 
         nt=nt, dim=dim, obs_input=obs_input, warm_nt=warm_nt, da_intv=da_intv)
print("Scalar Kernel Simulation complete. Results saved to PFF_scalar_results.npz")