# pfpf/initialization.py
import numpy as np
from typing import Dict, Any, Tuple


def initialization_filter(ps: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Initializes filter working variables (vg) and output containers.

    Parameters
    ----------
    ps : dict
        Parameter structure. Must contain:
            - "setup" with fields T, dimState_all, nParticle, PFPF, pf_type (optional)
            - "initparams" with field "initfcn": callable (ps, nParticle) -> (xp, M, PU)

    Returns
    -------
    vg : dict
        Working variables for the filter.
    output : dict
        Filter outputs (x_est, x_est_unweighted, etc.).
    """
    setup = ps["setup"]
    T = setup.T
    dim = setup.dimState_all
    n_particle = setup.nParticle

    # Output containers
    output = {
        "x_est": np.zeros((dim, T)),
        "x_est_unweighted": np.zeros((dim, T)),
    }

    vg: Dict[str, Any] = {}

    # Kalman variables
    vg["M"] = np.zeros((dim, 1))              # EKF/UKF estimate
    vg["PP"] = np.zeros((dim, dim))           # Predicted covariance
    vg["PU"] = np.zeros((dim, dim))           # Updated covariance

    # Particle initialization: xp, M, PU
    initfcn = ps["initparams"]["initfcn"]
    if not callable(initfcn):
        raise TypeError("ps['initparams']['initfcn'] must be a callable, not a string.")

    xp, M, PU = initfcn(ps, setup.nParticle)
    vg["xp"] = np.asarray(xp)          # (dim, N)
    vg["M"] = np.asarray(M).reshape(dim, 1)
    vg["PU"] = np.asarray(PU)

    # In PF-PF (LEDH), copy PU to PU_all to break correlations between particles
    if setup.PFPF and hasattr(setup, "pf_type"):
        if setup.pf_type == "LEDH":
            vg["PU_all"] = np.repeat(vg["PU"][:, :, None], n_particle, axis=2)

    vg["xp_m"] = vg["M"]                                  # current state estimate
    vg["logW"] = np.zeros(n_particle)                     # log-weights

    return vg, output

from .particle_flow import particle_estimate, cov_regularize, update_measurement_cov
from .particle_flow import stratified_resample  # if you want; not used here
from .particle_flow import particle_flow        # not used here but often imported together

def ekf_predict1(M, P, u, Q, f, df_dx, params):
    """
    Simple linear EKF prediction step, matching Acoustic model.

    M : (dim,1) prior mean
    P : (dim,dim) prior covariance
    u : unused (kept for MATLAB signature compatibility)
    Q : (dim,dim) process noise covariance
    f : unused (we use params["Phi"] directly)
    df_dx : unused
    params : dict, must contain "Phi"

    Returns
    -------
    M_prior : (dim,1)
    P_prior : (dim,dim)
    """
    M = np.asarray(M).reshape(-1, 1)
    P = np.asarray(P)
    Q = np.asarray(Q)

    Phi = np.asarray(params["Phi"])  # state transition matrix

    M_prior = Phi @ M
    P_prior = Phi @ P @ Phi.T + Q

    return M_prior, P_prior


def ukf_predict1(M, P, f, Q, params):
    """
    For the linear Acoustic model, UKF prediction is identical
    to EKF/Kalman prediction, so we just reuse ekf_predict1.
    """
    # The ekf_predict1 signature has an extra 'u' and 'df_dx' argument,
    # which we don't use. We pass None for them.
    return ekf_predict1(M, P, None, Q, f, None, params)



def propagate_and_estimate_prior_covariance(
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    z_current: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Python translation of propagateAndEstimatePriorCovariance.m

    Samples particles from the prior and generates deterministic
    propagated particles for flow parameter calculation.

    Parameters
    ----------
    vg : dict
        Working variables.
    ps : dict
        Parameter structure.
    z_current : (zdim,) or (zdim,1) array
        Measurement at current time step (used only if clustering is enabled).

    Returns
    -------
    vg : dict
        Updated working variables.
    ps : dict
        Possibly updated parameter structure (e.g., likeparams.R).
    """
    z_current = np.asarray(z_current).reshape(-1, 1)

    propparams = ps["propparams"]
    setup = ps["setup"]
    tt = propparams["time_step"]

    # Create a "no-noise" propagation parameter set
    propparams_no_noise = dict(propparams)
    if setup.example_name == "Acoustic":
        propparams_no_noise["Q"] = 0 * np.asarray(propparams["Q"])
    elif setup.example_name == "Septier16":
        propparams_no_noise["W"] = 0
    else:
        raise ValueError("Example name does not match any known case.")

    propagatefcn = propparams["propagatefcn"]

    # === Estimate prior mean mu_0 from Kalman mean ===
    xp_det, _ = propagatefcn(vg["M"], propparams_no_noise)  # deterministic if Q=0
    vg["mu_0"] = xp_det

    # Update measurement covariance based on current mean
    ps = update_measurement_cov(vg["M"], ps)

    pf_type = setup.pf_type

    # === Estimate prior covariance(s) using EKF/UKF or sample covariance ===
    if pf_type == "EDH":
        if setup.kflag == "UKF1":
            vg["M_prior"], vg["PP"] = ukf_predict1(
                vg["M"],
                vg["PU"],
                propparams_no_noise["propagatefcn"],
                propparams["Q"],
                propparams_no_noise,
            )
        else:  # default EKF1
            vg["M_prior"], vg["PP"] = ekf_predict1(
                vg["M"],
                vg["PU"],
                None,
                propparams["Q"],
                propparams_no_noise["propagatefcn"],
                None,
                propparams_no_noise,
            )

        # Regularize PP if needed
        try:
            np.linalg.cholesky(vg["PP"])
        except np.linalg.LinAlgError:
            vg["PP"] = cov_regularize(vg["PP"])

    elif pf_type in ("LEDH", "LEDH_cluster"):
        xp = np.asarray(vg["xp"])
        n_particle = xp.shape[1]

        vg["M_prior_all"] = np.zeros_like(xp)
        vg["PP_all"] = np.zeros((xp.shape[0], xp.shape[0], n_particle))

        for i in range(n_particle):
            M_i = xp[:, i:i+1]
            PU_i = vg["PU_all"][:, :, i]

            if setup.kflag == "UKF1":
                M_prior_i, PP_i = ukf_predict1(
                    M_i,
                    PU_i,
                    propparams_no_noise["propagatefcn"],
                    propparams["Q"],
                    propparams_no_noise,
                )
            else:  # EKF1
                M_prior_i, PP_i = ekf_predict1(
                    M_i,
                    PU_i,
                    None,
                    propparams["Q"],
                    propparams_no_noise["propagatefcn"],
                    None,
                    propparams_no_noise,
                )

            vg["M_prior_all"][:, i] = M_prior_i.reshape(-1)
            vg["PP_all"][:, :, i] = PP_i

            try:
                np.linalg.cholesky(vg["PP_all"][:, :, i])
            except np.linalg.LinAlgError:
                vg["PP_all"][:, :, i] = cov_regularize(vg["PP_all"][:, :, i])
    else:
        print("Warning: unspecified flow type in propagate_and_estimate_prior_covariance")

    # === Propagate particles with and without process noise ===
    xp_det_all, _ = propagatefcn(vg["xp"], propparams_no_noise)
    xp_prop_all, _ = propagatefcn(vg["xp"], propparams)

    vg["xp_prop_deterministic"] = xp_det_all
    vg["xp_prop"] = xp_prop_all

    # Auxiliary particles used for linearization in EDH/LEDH
    if pf_type == "EDH":
        vg["xp_auxiliary_individual"] = vg["mu_0"]
    elif pf_type in ("LEDH", "LEDH_cluster"):
        vg["mu_0_all"], _ = propagatefcn(vg["xp"], propparams_no_noise)
        vg["xp_auxiliary_individual"] = vg["xp_prop_deterministic"]

    # Optional clustering for LEDH_cluster
    if setup.use_cluster and pf_type == "LEDH_cluster":
        # perform_clustering is not yet implemented; only needed if you use clustering
        raise NotImplementedError("perform_clustering is not implemented")

    # Save previous particles and update current
    vg["xp_prev"] = vg["xp"]
    vg["xp"] = vg["xp_prop"]

    # State estimate from particle set
    vg["xp_m"], _ = particle_estimate(
        vg["logW"],
        vg["xp"],
        setup.maxilikeSAP,
        setup.maxilikemode,
    )

    if setup.doplot and setup.plotfcn is not None:
        setup.plotfcn(vg, ps, np.zeros_like(vg["xp"]), tt, "Prior")

    return vg, ps
