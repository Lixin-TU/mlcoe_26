# pfpf/acoustic_init.py

import numpy as np
from typing import Dict, Any, Tuple


def acoustic_gauss_init(ps: Dict[str, Any], n_particle: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize particles from a Gaussian around the initial state x0
    with diagonal covariance diag(sigma0.^2).

    Parameters
    ----------
    ps : dict
        Parameter structure. Must contain:
            - ps["initparams"]["x0"]      : (dim, 1) or (dim,) initial state
            - ps["initparams"]["sigma0"]  : (dim, 1) or (dim,) std for each state component
            - ps["setup"].dimState_all    : int, total state dimension
    n_particle : int
        Number of particles to draw.

    Returns
    -------
    xp : (dim, n_particle) array
        Initialized particles.
    M : (dim, 1) array
        Initial Kalman mean (set to x0).
    PU : (dim, dim) array
        Initial Kalman covariance (diag(sigma0.^2)).
    """
    dim = ps["setup"].dimState_all

    x0 = np.asarray(ps["initparams"]["x0"]).reshape(dim, 1)       # (dim,1)
    sigma0 = np.asarray(ps["initparams"]["sigma0"]).reshape(dim)  # (dim,)

    # Diagonal covariance
    PU = np.diag(sigma0**2)  # (dim, dim)

    # Draw particles: x0 + diag(sigma0) * N(0,1)
    # diag(sigma0) is represented via broadcasting
    noise = np.random.randn(dim, n_particle)
    xp = x0 + sigma0.reshape(dim, 1) * noise  # (dim, n_particle)

    M = x0.copy()  # initial mean is x0

    return xp, M, PU
