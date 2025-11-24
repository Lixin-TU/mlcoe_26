import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Need H_linear
sys.path.append(os.path.join(os.path.dirname(__file__), '../subroutines'))
from H_linear import H_linear

data = np.load('PFF_results.npz')
X = data['X']
Xt = data['Xt']
XnoDA = data['XnoDA']
nt = int(data['nt'])
warm_nt = int(data['warm_nt'])
obs_input = data['obs_input']
da_intv = int(data['da_intv'])

# obs space dimensions
ny_obs = len(obs_input)
np_particles = X.shape[1]

YY = np.zeros((nt, ny_obs, np_particles))
YY_noDA = np.zeros((nt, ny_obs, np_particles))
YY_true = np.zeros((nt, ny_obs))

inner_domain = [[idx] for idx in obs_input]

for i in range(nt):
    for j in range(ny_obs):
        inner_ind = inner_domain[j]
        
        # Modify to match default PFF.py (H_linear)
        YY[i, j, :] = H_linear(X[inner_ind, :, i].reshape(-1, np_particles))
        YY_noDA[i, j, :] = H_linear(XnoDA[inner_ind, :, i].reshape(-1, np_particles))
        YY_true[i, j] = H_linear(Xt[inner_ind, warm_nt + i].reshape(-1, 1))

ens_mean = np.mean(YY, axis=2)
ens_mean_noDA = np.mean(YY_noDA, axis=2)

RMSE = np.zeros(nt)
RMSE_noDA = np.zeros(nt)
spread = np.zeros(nt)
spread_noDA = np.zeros(nt)

for t in range(nt):
    RMSE_noDA[t] = np.sqrt(np.mean((ens_mean_noDA[t, :] - YY_true[t, :])**2))
    RMSE[t] = np.sqrt(np.mean((ens_mean[t, :] - YY_true[t, :])**2))
    
    spread[t] = np.mean(np.std(YY[t, :, :], axis=1))
    spread_noDA[t] = np.mean(np.std(YY_noDA[t, :, :], axis=1))

plt.figure(figsize=(10, 6))
plt.plot(range(nt), spread, color=[0.8, 0.2, 0.2], linewidth=2.5, label='spread (DA)')
plt.plot(range(nt), RMSE, color=[0.3, 0.3, 0.3], linewidth=2.5, label='RMSE (DA)')
plt.plot(range(nt), spread_noDA, '-.', color=[0.8, 0.2, 0.2], linewidth=2.5, label='spread (noDA)')
plt.plot(range(nt), RMSE_noDA, '-.', color=[0.3, 0.3, 0.3], linewidth=2.5, label='RMSE (noDA)')

plt.legend(fontsize=12, loc='upper left')
plt.grid(True)
plt.axis([0, nt, 0, 3])
plt.xlabel('timestep')
plt.title('RMSE obs space (linear identity)')
plt.show()