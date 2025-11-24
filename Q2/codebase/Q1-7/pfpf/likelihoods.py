# pfpf/likelihoods.py

import numpy as np
from typing import Dict, Any


def gaussian_llh(xp: np.ndarray, z: np.ndarray, likeparams: Dict[str, Any]) -> np.ndarray:
    """
    Computes log-likelihood of measurements z given particles xp under
    a Gaussian measurement noise model:
        z ~ N(h(x), R)

    Parameters
    ----------
    xp : (dim, N) array
        Particles.
    z : (m, 1) or (m,) array
        Measurement at current time step.
    likeparams : dict
        Must contain:
            - "h_func" : callable, h(x, likeparams) -> (m, N) or (m,1)
            - "R"      : (m,m) array  (or (m,m,N) for state-dependent R, not used in Acoustic)

    Returns
    -------
    llh : (N,) array
        Log-likelihood for each particle.
    """
    xp = np.asarray(xp)
    z = np.asarray(z).reshape(-1, 1)  # (m,1)

    h_func = likeparams["h_func"]
    R = np.asarray(likeparams["R"])

    # Evaluate measurement function h(x)
    # We expect h to be (m, N)
    h_x = np.asarray(h_func(xp, likeparams))  # (m, N)

    m, N = h_x.shape

    # For Acoustic example, R is constant (m,m)
    if R.ndim == 2:
        Rinv = np.linalg.inv(R)
        sign, logdetR = np.linalg.slogdet(R)
        if sign <= 0:
            raise ValueError("R must be positive definite for Gaussian likelihood.")
        const_term = -0.5 * (m * np.log(2 * np.pi) + logdetR)

        llh = np.zeros(N)
        for i in range(N):
            innov = z - h_x[:, i:i+1]  # (m,1)
            quad = float(innov.T @ Rinv @ innov)
            llh[i] = const_term - 0.5 * quad
    else:
        # If you later use Septier16 with state-dependent R(:,:,i),
        # you can extend this branch accordingly.
        raise NotImplementedError("State-dependent R not implemented in gaussian_llh.")

    return llh
