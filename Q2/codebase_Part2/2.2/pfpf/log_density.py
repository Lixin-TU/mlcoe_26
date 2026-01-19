import numpy as np


def log_gaussian_pdf(x: np.ndarray, m: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Multivariate Gaussian log-pdf for multiple samples.

    Parameters
    ----------
    x : (dim, N)
        Evaluation points.
    m : (dim, N) or (dim,1)
        Mean vectors.
    Q : (dim, dim)
        Covariance matrix.

    Returns
    -------
    logpdf : (N,)
    """
    x = np.asarray(x)
    m = np.asarray(m)

    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[1] == 1:
        m = m @ np.ones((1, x.shape[1]))  # replicate for all particles

    dim, N = x.shape
    Q = np.asarray(Q)

    # Precompute
    L = np.linalg.cholesky(Q)
    Qinv_xm = np.linalg.solve(L.T, np.linalg.solve(L, x - m))
    quad = np.sum((x - m) * Qinv_xm, axis=0)

    logdetQ = 2.0 * np.sum(np.log(np.diag(L)))

    logpdf = -0.5 * (dim * np.log(2 * np.pi) + logdetQ + quad)
    return logpdf


def log_proposal_density(vg: dict,
                         ps: dict,
                         log_jacobian_det_sum: np.ndarray) -> np.ndarray:
    """
    Python translation of log_proposal_density.m

    Computes the proposal log-density of particles after the particle flow.

    Parameters
    ----------
    vg : dict
        Working variables (xp_prop, xp_prop_deterministic, etc.)
    ps : dict
        Parameter structure
    log_jacobian_det_sum : (N,) array
        Sum of log Jacobians accumulated across lambda steps

    Returns
    -------
    log_proposal : (N,) array
    """
    xp_prop = np.asarray(vg["xp_prop"])                     # (dim, N)
    xp_det = np.asarray(vg["xp_prop_deterministic"])        # (dim, N or 1)

    example_name = ps["setup"].example_name

    if example_name == "Septier16":
        # MATLAB: computeGH_log_density(...)
        # You should provide computeGH_log_density.m so I can translate it.
        raise NotImplementedError(
            "computeGH_log_density is not implemented. Please provide the MATLAB file."
        )
    else:
        # Use Gaussian proposal: loggausspdf(x | mean=xp_det, cov=Q)
        Q = ps["propparams"]["Q"]
        log_proposal = log_gaussian_pdf(xp_prop, xp_det, Q)

    # MATLAB: log_proposal = log_proposal - log_jacobian_det_sum;
    return log_proposal - log_jacobian_det_sum


def update_measurement_cov(xp: np.ndarray, ps: dict) -> dict:
    """
    Python translation of updateMeasurementCov.m

    Computes state-dependent measurement covariance for the Septier16 example.

    Parameters
    ----------
    xp : (dim, N) array
        Particle set.
    ps : dict
        Parameter structure.

    Returns
    -------
    ps : dict
        Updated parameter structure with modified ps["likeparams"]["R"]
    """
    example_name = ps["setup"].example_name

    # Only applies to Septier16
    if example_name == "Septier16":
        h_func = ps["likeparams"]["h_func"]

        # h(xp) → (nMeasurement, N)
        cov_all_particles = np.asarray(h_func(xp, ps["likeparams"]))
        n_meas, N = cov_all_particles.shape

        # R will be (nMeasurement, nMeasurement, N)
        R_all = np.zeros((n_meas, n_meas, N))

        for i in range(N):
            R_all[:, :, i] = np.diag(cov_all_particles[:, i])

        ps["likeparams"]["R"] = R_all

    return ps
