import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('PFF_results.npz')
X = data['X']
Xt = data['Xt']
XnoDA = data['XnoDA']
nt = data['nt']
dim = int(data['dim'])
warm_nt = int(data['warm_nt'])
obs_input = data['obs_input'] # indices of observed vars

# Setup indices
all_ind = np.arange(dim)
obs_ind = obs_input
non_obs = np.setdiff1d(all_ind, obs_ind)
len_obs = len(obs_ind)
len_nobs = len(non_obs)

title_name = 'RMSE in state space'

XX = X
XXt = Xt
XXnoDA = XnoDA

# Calc RMSE
ens_mean = np.mean(XX, axis=1) # dim x nt
ens_mean_noDA = np.mean(XXnoDA, axis=1)

RMSE_noDA = np.zeros(nt)
RMSE_noDA_o = np.zeros(nt)
RMSE_noDA_n = np.zeros(nt)
RMSE = np.zeros(nt)
RMSE_o = np.zeros(nt)
RMSE_n = np.zeros(nt)

# Spread
spread = np.zeros(nt)
spread_noDA = np.zeros(nt)

for t in range(nt):
    # Time index for Xt is warm_nt + t (assuming Xt includes warm up)
    # Python 0-based indexing for arrays
    
    truth = XXt[:, warm_nt + t]
    
    RMSE_noDA[t]   = np.sqrt(np.mean((ens_mean_noDA[:, t] - truth)**2))
    RMSE_noDA_o[t] = np.sqrt(np.mean((ens_mean_noDA[obs_ind, t] - truth[obs_ind])**2))
    RMSE_noDA_n[t] = np.sqrt(np.mean((ens_mean_noDA[non_obs, t] - truth[non_obs])**2))
    
    RMSE[t]        = np.sqrt(np.mean((ens_mean[:, t] - truth)**2))
    RMSE_o[t]      = np.sqrt(np.mean((ens_mean[obs_ind, t] - truth[obs_ind])**2))
    RMSE_n[t]      = np.sqrt(np.mean((ens_mean[non_obs, t] - truth[non_obs])**2))
    
    # Spread
    spread[t] = np.mean(np.std(XX[:, :, t], axis=1))
    spread_noDA[t] = np.mean(np.std(XXnoDA[:, :, t], axis=1))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(nt), RMSE_o, 'r', linewidth=2, label='RMSE (obs)')
plt.plot(range(nt), RMSE_n, 'b', linewidth=2, label='RMSE (unobs)')
plt.plot(range(nt), RMSE_noDA_o, 'r-.', linewidth=2, label='RMSE (noDA obs)')
plt.plot(range(nt), RMSE_noDA_n, 'b-.', linewidth=2, label='RMSE (noDA unobs)')

plt.legend(fontsize=12, loc='upper left')
plt.grid(True)
plt.xlim([0, nt])
plt.ylim([0, 6])
plt.xlabel('timestep')
plt.title(title_name)
plt.tight_layout()
plt.show()