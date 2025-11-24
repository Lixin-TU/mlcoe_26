import numpy as np
from typing import Dict, Any, Tuple
from .log_density import log_gaussian_pdf, log_proposal_density, update_measurement_cov

# ============================================================
# Helpers that must be implemented separately
# (based on other MATLAB files: updateMeasurementCov.m, etc.)
# ============================================================

def particle_estimate(
    log_weights: np.ndarray,
    particles: np.ndarray,
    maxilikeSAP: int,
    maxilikemode: str,
    approxexpflag: bool = False,
) -> np.ndarray:
    """

    Forms an estimate based on a weighted set of particles.

    Parameters
    ----------
    log_weights : (N,) or (N,1) array
        Logarithmic weights of the particles.
    particles : (dim, N) array
        State values of the particles (state dimension x number of particles).
    maxilikeSAP : int
        Number of particles used in subspace-based approximation (unused here).
    maxilikemode : str
        'm' = median, 'a' = weighted mean (only weighted mean is implemented here).
    approxexpflag : bool
        If True, use polynomial approximation to exp (unused here).

    Returns
    -------
    estimate : (dim, 1) array
        Weighted mean of the particles.
    ml_weights : (N, 1) array
        Normalized weights in linear scale.
    """
    log_weights = np.asarray(log_weights).reshape(-1)
    particles = np.asarray(particles)

    # Subtract max log-weight for numerical stability
    log_weights = log_weights - np.max(log_weights)

    # Convert to linear weights
    ml_weights = np.exp(log_weights)
    ml_weights = ml_weights / np.sum(ml_weights)

    # Estimate = particles * weights
    # particles: (dim, N), ml_weights: (N,) -> (dim,)
    estimate = particles @ ml_weights

    # Return as column vector to mimic MATLAB (dim x 1)
    estimate = estimate.reshape(-1, 1)
    ml_weights = ml_weights.reshape(-1, 1)

    return estimate, ml_weights



import numpy as np

def log_process_density(vg: dict, ps: dict) -> np.ndarray:
    """
    Computes the process model log-density of the propagated particles.

    Parameters
    ----------
    vg : dict
        Working variables. Must contain:
            - "xp"                : (dim, N) propagated particles
            - "xp_prop_deterministic" : (dim, N) deterministic propagation (mean)
    ps : dict
        Parameter structure. Must contain:
            - ps["setup"].example_name
            - ps["propparams"]["Q"] : process noise covariance

    Returns
    -------
    log_process : (N,) array
        Log-density of each particle under the process model.
    """
    xp = np.asarray(vg["xp"])                    # (dim, N)
    xp_det = np.asarray(vg["xp_prop_deterministic"])  # (dim, N or 1)

    example_name = ps["setup"].example_name

    if example_name == "Septier16":
        # MATLAB: computeGH_log_density(vg.xp, vg.xp_prop_deterministic, ps.propparams)
        # This requires computeGH_log_density.m; not implemented yet.
        raise NotImplementedError(
            "computeGH_log_density is not implemented. Please provide the MATLAB file."
        )
    else:
        Q = ps["propparams"]["Q"]
        log_process = log_gaussian_pdf(xp, xp_det, Q)

    return log_process



def cov_regularize(cova: np.ndarray) -> np.ndarray:
    """
    Regularize a covariance matrix until it becomes positive definite
    under Cholesky factorization, or until a maximum number of iterations
    is reached.

    Parameters
    ----------
    cova : (dim, dim) array
        Covariance matrix to be regularized.

    Returns
    -------
    cova : (dim, dim) array
        Regularized covariance matrix.
    """
    cova = np.asarray(cova)
    dim = cova.shape[0]
    reg = np.eye(dim) * 1e-14

    count = 0
    max_count = 100

    while True:
        try:
            np.linalg.cholesky(cova)
            break  # positive definite
        except np.linalg.LinAlgError:
            if count >= max_count:
                # In MATLAB this only raises a warning; we mimic that behavior.
                print(
                    "Warning: cov_regularize:TooManyIterations - "
                    "Could not regularize the covariance matrix"
                )
                break
            cova = cova + reg
            count += 1

    return cova




def ekf_update1(M_prior, P_prior, z, dh_dx_func, R, h_func, u, likeparams):
    """
    EKF measurement update step.

    Parameters
    ----------
    M_prior : (dim,1) or (dim,) array
        Prior mean.
    P_prior : (dim,dim) array
        Prior covariance.
    z : (m,1) or (m,) array
        Current measurement.
    dh_dx_func : callable
        Function that computes measurement Jacobian H(x).
    R : (m,m) array
        Measurement noise covariance (for Acoustic, constant).
    h_func : callable
        Measurement function h(x).
    u : unused
        Kept for MATLAB compatibility.
    likeparams : dict
        Measurement model parameters (passed to h_func, dh_dx_func).

    Returns
    -------
    M_post : (dim,1) array
        Posterior mean.
    P_post : (dim,dim) array
        Posterior covariance.
    """
    # Ensure correct shapes
    M_prior = np.asarray(M_prior).reshape(-1, 1)
    P_prior = np.asarray(P_prior)
    z = np.asarray(z).reshape(-1, 1)
    R = np.asarray(R)

    # Measurement Jacobian H and predicted measurement h(M_prior)
    H = dh_dx_func(M_prior, likeparams)  # expected shape (m, dim)
    h = h_func(M_prior, likeparams)      # expected shape (m, 1)

    # Innovation
    innov = z - h                        # (m,1)

    # Innovation covariance S and Kalman gain K
    S = H @ P_prior @ H.T + R            # (m,m)
    K = P_prior @ H.T @ np.linalg.inv(S) # (dim,m)

    # Posterior mean and covariance
    M_post = M_prior + K @ innov
    P_post = P_prior - K @ H @ P_prior

    return M_post, P_post



def ukf_update1(
    M_prior: np.ndarray,
    P_prior: np.ndarray,
    z_current: np.ndarray,
    h_func,
    R: np.ndarray,
    likeparams: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python placeholder for ukf_update1.m.
    Should return (M_post, P_post).
    """
    raise NotImplementedError("ukf_update1 must be implemented from ukf_update1.m")


def stratified_resample(n_particle: int, weights: np.ndarray) -> np.ndarray:
    """
    Stratified resampling of particle indices.

    Parameters
    ----------
    n_particle : int
        Number of particles to resample.
    weights : (N,) array
        Normalized non-negative weights that sum to 1.

    Returns
    -------
    indices : (n_particle,) int array
        Indices of resampled particles (0-based).
    """
    weights = np.asarray(weights).reshape(-1)
    N = weights.size
    if N != n_particle:
        raise ValueError("weights length must equal n_particle")

    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # ensure numerical stability

    # Stratified samples: u_i ~ U((i-1)/N, i/N)
    u = (np.arange(N) + np.random.rand(N)) / float(N)

    indices = np.zeros(N, dtype=int)
    j = 0
    for i in range(N):
        while u[i] > cdf[j]:
            j += 1
        indices[i] = j
    return indices


def resample_regularize(vg: dict, ps: dict) -> dict:
    """
    Performs resampling (if needed) and adds regularization noise.

    Parameters
    ----------
    vg : dict
        Working variables. Must contain:
            - "logW"     : (N,) log-weights
            - "xp"       : (dim, N) particles
            - optionally "PU_all" : (dim, dim, N) for LEDH
    ps : dict
        Parameter structure. Must contain:
            - ps["setup"].nParticle
            - ps["setup"].Neff_thresh_ratio
            - ps["setup"].pf_type
            - ps["setup"].use_cluster
            - ps["setup"].regularize_resample
            - ps["propparams"]["Q_regularized"]
            - ps["propparams"]["time_step"]
            - ps["setup"].doplot
            - ps["setup"].plotfcn  (if plotting is used)

    Returns
    -------
    vg : dict
        Updated working variables after resampling and regularization.
    """
    tt = ps["propparams"]["time_step"]

    # Compute normalized weights from log-weights
    logW = np.asarray(vg["logW"]).reshape(-1)
    logW = logW - np.max(logW)
    weights = np.exp(logW)
    weights = weights / np.sum(weights)

    # Effective sample size
    eff = 1.0 / np.sum(weights**2)
    vg["eff"] = eff

    n_particle = ps["setup"].nParticle
    threshold = ps["setup"].Neff_thresh_ratio * n_particle

    # Decide whether to resample
    if ps["setup"].use_cluster or (eff < threshold):
        # Stratified resampling indices (0-based)
        I = stratified_resample(n_particle, weights)

        # Resample particles
        xp = np.asarray(vg["xp"])
        vg["xp"] = xp[:, I]

        # If we track individual covariances for LEDH, resample them too
        pf_type = ps["setup"].pf_type
        if pf_type in ("LEDH", "LEDH_cluster") and "PU_all" in vg:
            PU_all = np.asarray(vg["PU_all"])
            vg["PU_all"] = PU_all[:, :, I]

        # Reset log-weights after resampling
        vg["logW"] = np.zeros(n_particle)

        # Optional: plotting after resampling
        if ps["setup"].doplot and ps["setup"].plotfcn is not None:
            vg_tmp = dict(vg)
            vg_tmp["xp_m"] = np.mean(vg["xp"], axis=1, keepdims=True)
            ps["setup"].plotfcn(
                vg_tmp,
                ps,
                np.zeros_like(vg["xp"]),
                tt,
                "stratified resample",
            )

        # Regularization: add Gaussian noise with covariance Q_regularized
        if ps["setup"].regularize_resample:
            Q_reg = np.asarray(ps["propparams"]["Q_regularized"])
            dim = vg["xp"].shape[0]

            # Draw N samples from N(0, Q_reg)
            L = np.linalg.cholesky(Q_reg)
            z = np.random.randn(dim, n_particle)
            added_term = L @ z  # (dim, N)

            vg["xp"] = vg["xp"] + added_term

            if ps["setup"].doplot and ps["setup"].plotfcn is not None:
                ps["setup"].plotfcn(
                    vg,
                    ps,
                    np.zeros_like(vg["xp"]),
                    tt,
                    "regularized resample",
                )

    return vg



def homotopy_mean(
    z: np.ndarray,
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    lam: float,
    step_size: float,
) -> Dict[str, np.ndarray]:
    """
    Python translation of homotopy_Mean.m (EDH flow).

    Computes the flow slope using a single gradient evaluated at the mean.
    """
    xp = np.asarray(vg["xp"])
    dim, n_particles = xp.shape

    # For PFPF we use a deterministic flow starting from the prior mean.
    # In MATLAB: xp_m = xp_auxiliary_individual (which is actually the prior mean).
    if ps["setup"].PFPF:
        vg["xp_m"] = np.asarray(vg["xp_auxiliary_individual"])

    xp_m = np.asarray(vg["xp_m"]).reshape(dim, 1)

    likeparams = ps["likeparams"]

    # H: (zdim, dim), h: (zdim, 1)
    H = likeparams["dh_dx_func"](xp_m, likeparams)  # dh/dx at mean
    h = likeparams["h_func"](xp_m, likeparams)      # h at mean

    H = np.asarray(H)
    h = np.asarray(h).reshape(-1, 1)
    z = np.asarray(z).reshape(-1, 1)

    # e = h - H * xp_m
    e = h - H @ xp_m

    # zc = z - e
    zc = z - e  # (zdim, 1)

    # S = PP * H'
    PP = np.asarray(vg["PP"])
    S = PP @ H.T  # (dim, zdim)

    # A = -0.5 * S * inv(lambda * H * S + R) * H
    R = np.asarray(likeparams["R"])
    if R.ndim == 3:
        # If R is time-varying, you may need an index here.
        # For now we assume a single matrix.
        R = R[:, :, 0]
    M = lam * (H @ S) + R  # (zdim, zdim)
    middle = np.linalg.solve(M, H)  # (zdim, dim)
    A = -0.5 * S @ middle  # (dim, dim)

    I = np.eye(dim)
    # b = (I + 2*lambda*A) * ( (I+lambda*A)*S*inv(R)*(zc) + A*mu_0 )
    mu_0 = np.asarray(vg["mu_0"]).reshape(dim, 1)
    tmp_1 = np.linalg.solve(R, zc)        # (zdim, 1)
    tmp_2 = S @ tmp_1                     # (dim, 1)
    tmp_3 = (I + lam * A) @ tmp_2         # (dim, 1)
    tmp_4 = A @ mu_0                      # (dim, 1)
    inner = tmp_3 + tmp_4                 # (dim, 1)
    b = (I + 2.0 * lam * A) @ inner       # (dim, 1)

    # slope.real = A * xp + b * ones(1, N)
    ones_vec = np.ones((1, n_particles))
    slope_real = A @ xp + b @ ones_vec  # (dim, N)

    if ps["setup"].PFPF:
        slope_aux_ind = (A @ xp_m + b).reshape(dim, 1)
    else:
        slope_aux_ind = np.zeros((dim, 0))

    slope = {
        "real": slope_real,
        "auxiliary_individual": slope_aux_ind,
    }
    return slope


def homotopy_local(
    z: np.ndarray,
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    lam: float,
    step_size: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Python translation of homotopy_Local.m (LEDH flow).

    Computes the update for each particle using local gradients.
    Also returns the log determinant of Jacobian factors.
    """
    z = np.asarray(z).reshape(-1, 1)
    xp = np.asarray(vg["xp"])
    dim, n_particles = xp.shape
    zdim = z.shape[0]

    # Select linearization points
    if ps["setup"].PFPF:
        if ps["setup"].use_cluster:
            xp_linearization = np.asarray(vg["xp_auxiliary_cluster"])  # (dim, nLin)
        else:
            xp_linearization = np.asarray(vg["xp_auxiliary_individual"])
    else:
        xp_linearization = xp

    n_lin = xp_linearization.shape[1]

    likeparams = ps["likeparams"]

    # H: (zdim, dim, nLin), h: (zdim, nLin)
    H = likeparams["dh_dx_func"](xp_linearization, likeparams)
    h = likeparams["h_func"](xp_linearization, likeparams)

    H = np.asarray(H)                      # (zdim, dim, nLin)
    h = np.asarray(h).reshape(zdim, n_lin) # (zdim, nLin)

    # Compute error term e(:,i) = h(:,i) - H(:,:,i) * xp_linearization(:,i)
    e = np.zeros_like(h)
    for i in range(n_lin):
        Hi = H[:, :, i]                               # (zdim, dim)
        xi = xp_linearization[:, i].reshape(dim, 1)   # (dim, 1)
        e[:, i:i+1] = h[:, i:i+1] - Hi @ xi           # (zdim, 1)

    # zc = z - e  -> (zdim, nLin)
    zc = z - e
    # reshape to (zdim, 1, nLin) to match MATLAB's indexing style
    zc = zc.reshape(zdim, 1, n_lin)

    # Initialize log_jacobian_det as ones (as in MATLAB)
    log_jacobian_det = np.ones(n_lin)

    # Check whether we need to compute Jacobian determinants
    pf_type = ps["setup"].pf_type
    boolean_calc_det = ps["setup"].PFPF and ("LEDH" in pf_type)

    if boolean_calc_det and step_size is None:
        raise ValueError("step_size is required to calculate the Jacobian determinant")

    # Allocate A, b
    A_all = np.zeros((dim, dim, n_lin))
    b_all = np.zeros((dim, n_lin))

    if ps["setup"].use_cluster:
        slope_aux_cluster = np.zeros((dim, n_lin))
    else:
        slope_aux_ind = np.zeros((dim, n_particles))

    slope_real = np.zeros((dim, n_particles))

    # Loop over local linearization points
    for i in range(n_lin):
        if ps["setup"].PFPF:
            PP_i = np.asarray(vg["PP_all"][:, :, i])
            mu_0_i = np.asarray(vg["mu_0_all"][:, i]).reshape(dim, 1)
        else:
            PP_i = np.asarray(vg["PP"])
            mu_0_i = np.asarray(vg["mu_0"]).reshape(dim, 1)

        Hi = H[:, :, i]  # (zdim, dim)

        R = np.asarray(likeparams["R"])
        if R.ndim == 3:
            # If R is per-particle/time-varying
            Ri = R[:, :, i]
        else:
            Ri = R

        # PP_HiTranspose = PP * Hi'
        PP_HiT = PP_i @ Hi.T  # (dim, zdim)

        # A_i = -0.5 * PP_HiT * inv(lambda*Hi*PP_HiT + Ri) * Hi
        M = lam * (Hi @ PP_HiT) + Ri  # (zdim, zdim)
        middle = np.linalg.solve(M, Hi)  # (zdim, dim)
        A_i = -0.5 * PP_HiT @ middle     # (dim, dim)

        A_all[:, :, i] = A_i

        # b(:,i) = (I + 2*lambda*A_i) * ((I + lambda*A_i)*PP_HiT*(Ri\zc(:,1,i)) + A_i*mu_0_i)
        I = np.eye(dim)
        zc_i = zc[:, 0, i].reshape(zdim, 1)  # (zdim, 1)
        tmp_1 = np.linalg.solve(Ri, zc_i)     # (zdim, 1)
        tmp_2 = PP_HiT @ tmp_1                # (dim, 1)
        tmp_3 = (I + lam * A_i) @ tmp_2       # (dim, 1)
        tmp_4 = A_i @ mu_0_i                  # (dim, 1)
        inner = tmp_3 + tmp_4                 # (dim, 1)
        b_i = (I + 2.0 * lam * A_i) @ inner   # (dim, 1)

        b_all[:, i] = b_i[:, 0]

        if ps["setup"].PFPF:
            if ps["setup"].use_cluster:
                # For cluster centroids
                slope_aux_cluster[:, i] = (A_i @ xp_linearization[:, i].reshape(dim, 1) + b_i)[:, 0]
            else:
                # For each individual particle
                slope_aux_ind[:, i] = (A_i @ xp_linearization[:, i].reshape(dim, 1) + b_i)[:, 0]

        if not ps["setup"].use_cluster:
            # slope_real for the corresponding particle
            slope_real[:, i] = (A_i @ xp[:, i].reshape(dim, 1) + b_i)[:, 0]

        if boolean_calc_det:
            J = np.eye(dim) + step_size * A_i
            log_jacobian_det[i] = np.log(np.abs(np.linalg.det(J)))

    slope: Dict[str, np.ndarray] = {}

    if ps["setup"].PFPF:
        if ps["setup"].use_cluster:
            slope["auxiliary_cluster"] = slope_aux_cluster
        else:
            slope["auxiliary_individual"] = slope_aux_ind

    slope["real"] = slope_real

    # If clustering is used, map cluster-level A,b back to particle-level slopes
    if ps["setup"].use_cluster:
        xp_cluster_ix = np.asarray(vg["xp_cluster_ix"], dtype=int)  # indices
        A_full = A_all[:, :, xp_cluster_ix]  # (dim, dim, N)
        b_full = b_all[:, xp_cluster_ix]     # (dim, N)
        log_jacobian_det = log_jacobian_det[xp_cluster_ix]

        slope_real_full = np.zeros((dim, xp.shape[1]))
        for j in range(A_full.shape[2]):
            A_j = A_full[:, :, j]
            b_j = b_full[:, j].reshape(dim, 1)
            slope_real_full[:, j] = (A_j @ xp[:, j].reshape(dim, 1) + b_j)[:, 0]

        slope["real"] = slope_real_full

    return slope, log_jacobian_det


def calculate_slope(
    z_current: np.ndarray,
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    lam: float,
    step_size: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Python translation of calculateSlope.m.
    Selects homotopy_Local or homotopy_Mean based on ps["setup"].pf_type.
    """
    n_particle = ps["setup"].nParticle
    log_jacobian_det = np.zeros(n_particle)

    pf_type = ps["setup"].pf_type

    if pf_type in ("LEDH_cluster", "LEDH"):
        slope, log_jacobian_det = homotopy_local(
            z=z_current,
            vg=vg,
            ps=ps,
            lam=lam,
            step_size=step_size,
        )
    elif pf_type == "EDH":
        slope = homotopy_mean(
            z=z_current,
            vg=vg,
            ps=ps,
            lam=lam,
            step_size=step_size,
        )
    else:
        raise ValueError(f"Unknown pf_type: {pf_type}")

    return slope, log_jacobian_det



def particle_flow(
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    z_current: np.ndarray,
) -> Dict[str, Any]:
    """
    Python translation of particleFlow.m.
    Performs the particle flow and resampling (if needed).
    """
    tt = ps["propparams"]["time_step"]
    lambda_prev = 0.0

    n_particle = ps["setup"].nParticle
    log_jacobian_det_sum = np.zeros(n_particle)

    for lam in ps["setup"].lambda_range:
        step_size = float(lam - lambda_prev)

        # Update measurement covariance based on auxiliary particles (or mean)
        vg_aux = np.asarray(vg["xp_auxiliary_individual"])
        ps = update_measurement_cov(vg_aux, ps)

        slope, log_jacobian_det = calculate_slope(
            z_current=z_current,
            vg=vg,
            ps=ps,
            lam=lam,
            step_size=step_size,
        )

        log_jacobian_det_sum = log_jacobian_det_sum + log_jacobian_det
        log_jacobian_det_sum = log_jacobian_det_sum - np.max(log_jacobian_det_sum)

        if ps["setup"].use_cluster:
            vg["xp_auxiliary_cluster"] = (
                vg["xp_auxiliary_cluster"]
                + step_size * slope["auxiliary_cluster"]
            )
        else:
            vg["xp_auxiliary_individual"] = (
                vg["xp_auxiliary_individual"]
                + step_size * slope["auxiliary_individual"]
            )

        vg["xp"] = vg["xp"] + step_size * slope["real"]

        
        vg["xp_m"], ml_weights = particle_estimate(
            vg["logW"],
            vg["xp"],
            ps["setup"].maxilikeSAP,
            ps["setup"].maxilikemode,
        )


        lambda_prev = float(lam)

        if ps["setup"].doplot and ps["setup"].plotfcn is not None:
            title_str = (
                "particle flow, λ = "
                + f"{round(1e3 * lambda_prev) * 1e-3:g}"
            )
            ps["setup"].plotfcn(vg, ps, slope["real"], tt, title_str)

    vg["log_jacobian_det_sum"] = log_jacobian_det_sum

    vg = correction_and_calculate_weights(
        vg=vg,
        ps=ps,
        z_current=z_current,
        log_jacobian_det_sum=log_jacobian_det_sum,
    )

    return vg



def correction_and_calculate_weights(
    vg: Dict[str, Any],
    ps: Dict[str, Any],
    z_current: np.ndarray,
    log_jacobian_det_sum: np.ndarray,
) -> Dict[str, Any]:
    """
    Python translation of correctoinAndCalculateWeights.m.
    Calculates importance weights, updates covariance, and resamples.
    """
    z_current = np.asarray(z_current).reshape(-1, 1)

    log_prop = log_proposal_density(vg, ps, log_jacobian_det_sum)
    log_prior = log_process_density(vg, ps)

    likeparams = ps["likeparams"]
    llh = likeparams["llh"](vg["xp"], z_current, likeparams)

    vg["logW"] = log_prior + llh - log_prop + vg["logW"]
    vg["logW"] = vg["logW"] - np.max(vg["logW"])

    # Update state estimate from weighted particles
    vg["xp_m"], ml_weights = particle_estimate(
        vg["logW"],
        vg["xp"],
        ps["setup"].maxilikeSAP,
        ps["setup"].maxilikemode,
    )

    # Copy particle mean to Kalman mean
    vg["M"] = vg["xp_m"]

    # One step EKF/UKF covariance update at mean
    ps = update_measurement_cov(vg["M"], ps)

    pf_type = ps["setup"].pf_type
    example_name = ps["setup"].example_name
    kflag = ps["setup"].kflag

    if pf_type == "EDH":
        if kflag == "EKF1":
            _, vg["PU"] = ekf_update1(
                vg["M_prior"],
                vg["PP"],
                z_current,
                likeparams["dh_dx_func"],
                likeparams["R"],
                likeparams["h_func"],
                None,
                likeparams,
            )
        elif kflag == "UKF1":
            _, vg["PU"] = ukf_update1(
                vg["M_prior"],
                vg["PP"],
                z_current,
                likeparams["h_func"],
                likeparams["R"],
                likeparams,
            )
        elif kflag == "none":
            weights = np.exp(vg["logW"])
            weights = weights / np.sum(weights)
            de_mean = vg["xp"] - vg["xp_m"].reshape(-1, 1)
            vg["PU"] = (de_mean * weights) @ de_mean.T / (1.0 - np.sum(weights**2))
        else:
            raise ValueError(f"Unknown kflag: {kflag}")

        # Regularize covariance if needed
        try:
            np.linalg.cholesky(vg["PU"])
        except np.linalg.LinAlgError:
            vg["PU"] = cov_regularize(vg["PU"])

    elif pf_type in ("LEDH", "LEDH_cluster"):
        if example_name == "Septier16":
            ps = update_measurement_cov(vg["xp"], ps)
            R_likeparams = np.asarray(ps["likeparams"]["R"])
        else:
            R_likeparams = None

        n_particles = vg["xp"].shape[1]
        for i in range(n_particles):
            if example_name == "Septier16" and R_likeparams is not None:
                likeparams["R"] = R_likeparams[:, :, i]

            if kflag == "EKF1":
                _, vg["PU_all"][:, :, i] = ekf_update1(
                    vg["M_prior_all"][:, i],
                    vg["PP_all"][:, :, i],
                    z_current,
                    likeparams["dh_dx_func"],
                    likeparams["R"],
                    likeparams["h_func"],
                    None,
                    likeparams,
                )
            elif kflag == "UKF1":
                _, vg["PU_all"][:, :, i] = ukf_update1(
                    vg["M_prior_all"][:, i],
                    vg["PP_all"][:, :, i],
                    z_current,
                    likeparams["h_func"],
                    likeparams["R"],
                    likeparams,
                )
            else:
                raise ValueError(f"Unknown kflag: {kflag}")

            try:
                np.linalg.cholesky(vg["PU_all"][:, :, i])
            except np.linalg.LinAlgError:
                vg["PU_all"][:, :, i] = cov_regularize(vg["PU_all"][:, :, i])

    # Finally, resample and regularize
    vg = resample_regularize(vg, ps)

    return vg
