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


# for experiment setup:
DA_run   = 1                  # DA_run = 1: run the ensemble with DA cycles
noDA_run = 1                  # noDA_run = 1: run the ensemble without DA cycles
nt       = 200                # number of integration timestep
t_start  = 0                  # the (re)start time of the model (Python 0-based)
warm_nt  = 1000               # warm up time for the Lorenz model
gen_ens  = 1                  # generate_ens = 1: (for the first run)
np_particles = 30             # number of particles (renamed from np to avoid conflict with numpy)

# parameters for the L96 model:
dim = 40                      # dimension for lorenz 96 model
F   = 8.0                     # forcing 
dt  = 0.01                    # time resolution

# settings for generating the ensemble
Q     = 2 * np.eye(dim)       # background error covariance
Q_inv = np.linalg.inv(Q)

# settings for DA/PFF 
# /PFF kernel:
alpha    = 1.0/np_particles   # the tuning parameter for the covariance of the kernel
# /PFF iteration:
max_pseudo_step     = 150     # maximum number of iterations
eps_init            = 5e-2    # initial pseudo-timestep size (learning rate)  
stop_cri            = 1e-3    # when the norm(grad_KL(s)) < stop_cri, the PFF iteration stops
stop_cri_percentage = 0.05    # when norm(grad_KL(s)) < stop_cri_percentage*norm(grad_KL(1)), stop
min_learning_rate   = 1e-5    # when learning rate < min_learning rate, stop
# /PFF prior assumption:
io_local       = 1            # 0: no localization / 1: localization
r_influ        = 4            # localization "radius"
io_gauss_prior = 1            # 0 = gaussian mixture prior, 1 = gaussain prior
inflation_fac  = 1.25         # inflation factor for prior covariance
tune_C  = 5.0/inflation_fac   # make the covariance of the component of gaussian mixture larger
# /PFF precondition:
pre_cond  = 1                 # 0 = posterior covariance, 1 = prior covariance
# /SVD
cond_num  = -5                # condition number for SVD

# settings for observation
da_intv  = 20                 # obs frequency

# the inner domain for obs:
# linear identity obs
obs_den      = 2
# Python 0-based indexing: start at obs_den-1, step by obs_den
obs_input    = np.arange(obs_den-1, dim, obs_den) 
ny_obs = len(obs_input)
# Create inner_domain list (equivalent to cell array)
inner_domain = [[idx] for idx in obs_input]

# generate the observation error first:
np.random.seed(0) # Default seed behavior
obs_err   = 0.3
R         = obs_err**2 * np.eye(ny_obs)
total_obs = int(nt/da_intv)
# Gaussian obs error
obs_rnd   = np.random.multivariate_normal(np.zeros(ny_obs), R, total_obs).T

# declare matrices
if gen_ens == 1:
    prior = np.zeros((dim, np_particles, total_obs))
    
obs = np.zeros((ny_obs, total_obs))

norm_grad_KL = np.zeros((total_obs, max_pseudo_step))

# %% Integrate the L96 model (warm up):

Xt = np.zeros((dim, warm_nt + nt))
Xt[:, 0] = F * np.ones(dim)
# Perturbed IC: indexes 7, 15, 23, 31, 39 (MATLAB 8:8:40)
Xt[7::8, 0] = F + 1 

for t in range(warm_nt + nt - 1):
    Xt[:, t+1] = L96_RK4(Xt[:, t].reshape(-1, 1), dt, F).flatten()

# %% Generate the ensemble
if gen_ens == 1:
    ctlmean = Xt[:, warm_nt] + np.random.multivariate_normal(np.zeros(dim), np.eye(dim), 1).flatten()
    
    # initial condition
    X = np.zeros((dim, np_particles, nt))
    # mvnrnd(mu, sigma, n) -> n x dim. Transpose to dim x n
    X[:, :, 0] = np.random.multivariate_normal(ctlmean, Q, np_particles).T

# %% Integration of ensemble & Sequential Data Assimilation

if noDA_run == 1:
    XnoDA = np.zeros((dim, np_particles, nt))
    XnoDA[:, :, 0] = X[:, :, 0]
    for t in range(nt - 1):
        XnoDA[:, :, t+1] = L96_RK4(XnoDA[:, :, t], dt, F)

if DA_run == 1:
    t = t_start
    # Python loop from 0 to nt-1.
    # Logic adjustment: MATLAB loop `while t < nt`. Inside `X(:,:,t+1)`.
    # Current t is index of current state. Calculate next state t+1.
    while t < nt - 1:
        print(f'start timestep t={t+1}') # t+1 corresponds to MATLAB index for next step
        
        # Check DA time (MATLAB used t+1 because 1-based, here t+1 works as step count)
        if (t+1) % da_intv == 0:
            io_obs = 1
            io_pff = 1
            print(f'start DA at t={t+1}')
        else:
            io_obs = 0
            io_pff = 0
            
        # Step 1 -- run the model
        X[:, :, t+1] = L96_RK4(X[:, :, t], dt, F)
        
        # Step 2 -- Generate the observation
        if io_obs == 1:
            obs_time_idx = int((t+1)/da_intv) - 1 # 0-based index for obs storage
            for i in range(ny_obs):
                inner_ind = inner_domain[i]
                # Xt needs proper indexing. Xt has size warm_nt + nt.
                # MATLAB: Xt(inner_ind, warm_nt+t+1)
                # Python: Xt[inner_ind, warm_nt+t+1]
                val = Xt[inner_ind, warm_nt + t + 1]
                obs[i, obs_time_idx] = H_linear(val.reshape(-1,1)) + obs_rnd[i, obs_time_idx]

        # Step 3 -- Particle Flow Filter (PFF)
        if io_pff == 1:
            
            X_tmp = X[:, :, t+1]
            X_mean = np.tile(np.mean(X_tmp, axis=1, keepdims=True), (1, np_particles))
            
            # Covariance calculation
            diff = X_tmp - X_mean
            C = inflation_fac * (diff @ diff.T) / (np_particles - 1) / np_particles
            
            if io_local == 1:
                tmp = np.zeros((dim, dim))
                for i in range(1, 3 * r_influ + 1):
                    # Construct diagonals
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
            
            # Precondition
            if pre_cond == 0:
                # Not implemented fully as P_inv logic commented out in original
                pass 
            elif pre_cond == 1:
                qn = B / inflation_fac
                
            # Start iteration
            s = 0 # Python 0-based iteration index
            ct = 0
            norm_grad_KL[obs_time_idx, 0] = 1e8
            eps = eps_init * np.ones(max_pseudo_step)
            
            pseudo_X = np.zeros((dim, np_particles, max_pseudo_step))
            pseudo_X[:, :, 0] = X_mean + (X_tmp - X_mean)
            grad_KL = np.zeros((dim, np_particles, max_pseudo_step))
            
            while s < max_pseudo_step:
                if s > 0 and norm_grad_KL[obs_time_idx, max(s-1, 0)] <= stop_cri:
                    break
                    
                Hx_ens = np.zeros((ny_obs, np_particles))
                dHdx = np.zeros((ny_obs, dim, np_particles))
                
                for i in range(ny_obs):
                    inner_ind = inner_domain[i]
                    # obs operator
                    current_X_sub = pseudo_X[inner_ind, :, s].reshape(len(inner_ind), -1)
                    Hx_ens[i, :] = H_linear(current_X_sub).flatten()
                    
                    # adjoint
                    tmp_dHdx = H_linear_adjoint(current_X_sub)
                    # Broadcasting assignment
                    # inner_ind is list of ints.
                    for part_idx in range(np_particles):
                        for k, d_idx in enumerate(inner_ind):
                             dHdx[i, d_idx, part_idx] = tmp_dHdx[k, part_idx]
                
                tmp_grad_log_post = np.zeros((dim, np_particles))
                p_obs = np.zeros((dim, np_particles))
                p_bkg = np.zeros((dim, np_particles))
                
                # Gradient calculation
                for i in range(np_particles):
                    if io_gauss_prior == 0:
                        # GMM implementation omitted for brevity/risk, sticking to recommended io_gauss_prior=1
                        pass
                    elif io_gauss_prior == 1:
                        p_bkg[:, i] = -B_inv @ (pseudo_X[:, i, s] - np.mean(X_tmp, axis=1))
                        
                    HT = dHdx[:, :, i].T
                    p_obs[:, i] = HT @ (np.linalg.inv(R) @ (obs[:, obs_time_idx] - Hx_ens[:, i]))
                    tmp_grad_log_post[:, i] = p_obs[:, i] + p_bkg[:, i]
                    
                # Particle flow kernel
                tmp_K = np.zeros((dim, np_particles, np_particles))
                tmp_grad_K = np.zeros((dim, np_particles, np_particles))
                
                # Warning: Nested loops are slow in Python, but keeping strict translation
                for d_idx in range(dim):
                    for i in range(np_particles):
                        for j in range(np_particles):
                            if j >= i:
                                diff_val = pseudo_X[d_idx, i, s] - pseudo_X[d_idx, j, s]
                                val = np.exp(-0.5 * diff_val * (1.0/(B[d_idx, d_idx]*alpha)) * diff_val)
                                tmp_K[d_idx, i, j] = val
                                tmp_grad_K[d_idx, i, j] = -val / (B[d_idx, d_idx]*alpha) * (-diff_val)
                            else:
                                tmp_K[d_idx, i, j] = tmp_K[d_idx, j, i]
                                tmp_grad_K[d_idx, i, j] = -tmp_grad_K[d_idx, j, i]
                                
                            grad_KL[d_idx, i, s] += (tmp_K[d_idx, i, j] * tmp_grad_log_post[d_idx, j] + tmp_grad_K[d_idx, i, j]) / np_particles
                
                norm_val = np.sqrt(np.sum(grad_KL[:, :, s]**2) / (dim * np_particles))
                norm_grad_KL[obs_time_idx, s] = norm_val
                
                print(f"iteration s={s+1} norm ={norm_val/norm_grad_KL[obs_time_idx, 0]*100:.2f} eps = {eps[s]}")
                
                # Adaptive Learning Rate
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
                    
                    # Look back logic
                    prev_max = 0
                    if s > 0:
                        prev_max = np.max(norm_grad_KL[obs_time_idx, s-1:s]) # slice s-1:s is just s-1
                    
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
            prior[:, :, obs_time_idx] = X[:, :, t+1] # Save Prior
            X[:, :, t+1] = pseudo_X[:, :, s_end] # Save Analysis
            
        t += 1

# Save results for diagnostics
np.savez('PFF_results.npz', 
         X=X, Xt=Xt, XnoDA=XnoDA, prior=prior, 
         nt=nt, dim=dim, obs_input=obs_input, warm_nt=warm_nt, da_intv=da_intv)
print("Simulation complete. Results saved to PFF_results.npz")