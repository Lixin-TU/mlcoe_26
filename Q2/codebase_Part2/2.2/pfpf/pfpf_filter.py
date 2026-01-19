# pfpf/pfpf_filter.py
import time
import numpy as np
from typing import Dict, Any, Tuple

from .initialization import initialization_filter, propagate_and_estimate_prior_covariance
from .particle_flow import particle_flow
from .calculate_errors import calculateErrors   # you will translate this from calculateErrors.m


def PFPF(ps: Dict[str, Any], z: np.ndarray) -> Dict[str, Any]:
    """
    Implements the Particle Flow Particle Filter (PF-PF).

    Parameters
    ----------
    ps : dict
        Parameter structure.
    z : (zdim, T) array
        Measurement matrix, each column is one time step.

    Returns
    -------
    output : dict
        Contains x_est, x (true state), execution_time, Neff (optional), etc.
    """
    start_time = time.time()

    vg, output = initialization_filter(ps)

    z = np.asarray(z)
    T = z.shape[1]

    for tt in range(T):
        ps["propparams"]["time_step"] = tt + 1  # MATLAB is 1-based

        # Propagate particles and estimate prior covariance
        vg, ps = propagate_and_estimate_prior_covariance(vg, ps, z[:, tt:tt+1])

        # Particle flow update
        vg = particle_flow(vg, ps, z[:, tt:tt+1])

        # Store state estimate
        output["x_est"][:, tt:tt+1] = vg["xp_m"]

        # Effective sample size
        if "eff" in vg:
            if "Neff" not in output:
                output["Neff"] = np.zeros(T)
            output["Neff"][tt] = vg["eff"]

    output["x"] = ps["x"]
    output["execution_time"] = time.time() - start_time

    alg_name = f"PFPF_{ps['setup'].pf_type}"
    calculateErrors(output, ps, alg_name)

    return output
