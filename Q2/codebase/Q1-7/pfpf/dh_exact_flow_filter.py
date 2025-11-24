# pfpf/dh_exact_flow_filter.py
import time
import numpy as np
from typing import Dict, Any

from .initialization import initialization_filter
from .particle_flow import calculate_slope, cov_regularize, update_measurement_cov
from .particle_flow import particle_estimate
from .particle_flow import ekf_update1

from .calculate_errors import calculateErrors
from .initialization import ekf_predict1, ukf_predict1


def _sqrt_sym_pd(mat: np.ndarray) -> np.ndarray:
    """
    Symmetric matrix square root via eigen-decomposition.
    Used instead of scipy.linalg.sqrtm.
    """
    mat = np.asarray(mat)
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, 0, None)
    root = vecs @ np.diag(np.sqrt(vals_clipped)) @ vecs.T
    return root


def DH_ExactFlow_Filter(ps: Dict[str, Any], z: np.ndarray) -> Dict[str, Any]:
    """
    Implementation of the Daum-Huang exact flow particle filter
    (EDH/LEDH without PF-PF outer structure).

    Parameters
    ----------
    ps : dict
        Parameter structure.
    z : (zdim, T) array
        Measurements.

    Returns
    -------
    output : dict
        Contains x_est, x (true state), execution_time, etc.
    """
    start_time = time.time()
    setup = ps["setup"]
    T = setup.T

    vg, output = initialization_filter(ps)
    z = np.asarray(z)

    for tt in range(1, T + 1):
        # Optional redraw step
        if tt != 1 and setup.Redraw:
            dim = setup.dimState_all
            n_particle = setup.nParticle
            sqrt_PU = _sqrt_sym_pd(vg["PU"])
            noise = np.random.randn(dim, n_particle)
            vg["xp"] = sqrt_PU @ noise + vg["xp_m"]

        ps["propparams"]["time_step"] = tt

        # Propagate mean without noise
        propparams_no_noise = dict(ps["propparams"])
        if setup.example_name == "Acoustic":
            propparams_no_noise["Q"] = 0 * np.asarray(ps["propparams"]["Q"])
        elif setup.example_name == "Septier16":
            propparams_no_noise["W"] = 0
        else:
            raise ValueError("Example name does not match any known case.")

        propagatefcn = ps["propparams"]["propagatefcn"]

        # Kalman prediction step to get M_prior, PP
        ps = update_measurement_cov(vg["M"], ps)

        if setup.kflag == "EKF1":
            vg["M_prior"], vg["PP"] = ekf_predict1(
                vg["M"],
                vg["PU"],
                None,
                ps["propparams"]["Q"],
                propparams_no_noise["propagatefcn"],
                None,
                propparams_no_noise,
            )
        elif setup.kflag == "UKF1":
            vg["M_prior"], vg["PP"] = ukf_predict1(
                vg["M"],
                vg["PU"],
                propparams_no_noise["propagatefcn"],
                ps["propparams"]["Q"],
                propparams_no_noise,
            )
        elif setup.kflag == "none":
            xp_tmp, _ = propagatefcn(vg["xp"], ps["propparams"])
            vg["M_prior"] = np.mean(xp_tmp, axis=1, keepdims=True)
            vg["PP"] = np.cov(xp_tmp, bias=False)
        else:
            raise ValueError(f"Unknown kflag: {setup.kflag}")

        # Initial log-weights (uniform)
        logW = np.log(np.ones(setup.nParticle) / setup.nParticle)

        if setup.doplot and setup.plotfcn is not None:
            setup.plotfcn(vg, ps, np.zeros_like(vg["xp"]), tt, "before propagation")

        # Prior propagation with noise
        vg["xp"], _ = propagatefcn(vg["xp"], ps["propparams"])

        if setup.kflag == "none":
            vg["PP"] = np.cov(vg["xp"], bias=False)

        # State estimate from particles
        vg["xp_m"], _ = particle_estimate(logW, vg["xp"], setup.maxilikeSAP, setup.maxilikemode)
        vg["mu_0"] = vg["xp_m"]

        if setup.doplot and setup.plotfcn is not None:
            setup.plotfcn(vg, ps, np.zeros_like(vg["xp"]), tt, "after prior propagation")

        # Flow in lambda
        lambda_prev = 0.0
        for lam in setup.lambda_range:
            # Update measurement covariance based on EDH/LEDH
            if setup.pf_type in ("LEDH_cluster", "LEDH"):
                ps = update_measurement_cov(vg["xp"], ps)
            elif setup.pf_type == "EDH":
                ps = update_measurement_cov(vg["xp_m"], ps)

            step_size = float(lam - lambda_prev)

            slope_struct, _ = calculate_slope(
                z_current=z[:, tt - 1:tt],
                vg=vg,
                ps=ps,
                lam=lam,
                step_size=step_size,
            )
            slope = slope_struct["real"]

            if setup.doplot and setup.plotfcn is not None:
                title_str = f"in the process of particle flow, lambda = {lam:g}"
                setup.plotfcn(vg, ps, slope, tt, title_str)

            vg["xp"] = vg["xp"] + step_size * slope
            vg["xp_m"], _ = particle_estimate(logW, vg["xp"], setup.maxilikeSAP, setup.maxilikemode)
            lambda_prev = lam

        if setup.doplot and setup.plotfcn is not None:
            setup.plotfcn(vg, ps, np.zeros_like(vg["xp"]), tt, "after particle flow")

        # Final estimate and covariance update
        vg["xp_m"], _ = particle_estimate(logW, vg["xp"], setup.maxilikeSAP, setup.maxilikemode)
        vg["M"] = vg["xp_m"]


        # Update measurement covariance based on posterior mean
        ps = update_measurement_cov(vg["M"], ps)

        likeparams = ps["likeparams"]

        if setup.kflag == "EKF1":
            # EKF measurement update: use our ekf_update1
            R = likeparams["R"]
            _, vg["PU"] = ekf_update1(
                vg["M_prior"],               # prior mean
                vg["PP"],                    # prior covariance
                z[:, tt - 1:tt],             # current measurement column (m,1)
                likeparams["dh_dx_func"],    # Jacobian function
                R,                           # measurement noise covariance
                likeparams["h_func"],        # measurement function
                None,                        # u (unused)
                likeparams,                  # likeparams struct
            )
        elif setup.kflag == "UKF1":

            R = likeparams["R"]
            _, vg["PU"] = ekf_update1(
                vg["M_prior"],
                vg["PP"],
                z[:, tt - 1:tt],
                likeparams["dh_dx_func"],
                R,
                likeparams["h_func"],
                None,
                likeparams,
            )
        elif setup.kflag == "none":
            vg["PU"] = np.cov(vg["xp"], bias=False)


        # Regularize PU if needed
        try:
            np.linalg.cholesky(vg["PU"])
        except np.linalg.LinAlgError:
            vg["PU"] = cov_regularize(vg["PU"])

        # Store estimate
        output["x_est"][:, tt - 1] = vg["xp_m"].reshape(-1)

    output["x"] = ps["x"]
    output["execution_time"] = time.time() - start_time

    alg_name = setup.pf_type
    calculateErrors(output, ps, alg_name)

    return output
