import numpy as np
from typing import Dict, Any, Tuple


# =========================================================
# AcousticPropagate.m
# =========================================================
def acoustic_propagate(xp, prop_params):
    """

    xp : (dim, N) or (dim,) array
    prop_params : dict with keys 'Phi', 'Q', ...
    """
    Phi = prop_params["Phi"]
    Q = np.asarray(prop_params["Q"])

    xp = np.asarray(xp)
    if xp.ndim == 1:
        xp = xp[:, None]
    dim, N = xp.shape

    # If Q is (numerically) zero, do deterministic propagation
    if np.allclose(Q, 0):
        xp_new = Phi @ xp
    else:
        # Standard Gaussian process noise
        L = np.linalg.cholesky(Q)   # (dim, dim)
        noise = L @ np.random.randn(dim, N)
        xp_new = Phi @ xp + noise

    logwt = np.ones(N)
    return xp_new, logwt



# =========================================================
# Acoustic_hfunc.m
# =========================================================

def acoustic_hfunc(xp: np.ndarray, likeparams: Dict[str, Any]) -> np.ndarray:
    """
    Python version of Acoustic_hfunc.m

    Measurement function for the acoustic model.

    Inputs
    ------
    xp : (dim, nParticles) ndarray
        State particles.
    likeparams : dict
        Must contain keys:
            'sensorsPos' : (2*nTarget, nSensor)
            'Amp'        : scalar
            'd0'         : scalar
            'invPow'     : scalar
            'nTarget'    : int
            'nSensor'    : int

    Returns
    -------
    y : (nSensor, nParticles) ndarray
    """
    xp = np.asarray(xp)
    dim, n_particles = xp.shape

    n_sensor = int(likeparams["nSensor"])
    n_target = int(likeparams["nTarget"])
    sensors_pos = np.asarray(likeparams["sensorsPos"])  # (2*nTarget, nSensor)

    # Positions (x,y) for each target
    # xx = xp(1:4:nTarget*4,:);  xy = xp(2:4:nTarget*4,:)
    idx_x = np.arange(0, 4 * n_target, 4)
    idx_y = np.arange(1, 4 * n_target, 4)
    xx = xp[idx_x, :]  # (nTarget, nParticles)
    xy = xp[idx_y, :]  # (nTarget, nParticles)

    # Stack [x; y] -> (2*nTarget, nParticles)
    x = np.vstack((xx, xy))

    if n_particles > 1:
        # sensorsPos: (2*nTarget, nSensor)
        # x:         (2*nTarget, nParticles)
        #
        # v = sensorsPos - x (broadcast over particles)
        # result: (2*nTarget, nSensor, nParticles)
        v = sensors_pos[:, :, None] - x[:, None, :]
        v = v ** 2

        # Sum squared differences for x,y of each target
        # v(1:nTarget,:,:) + v(nTarget+1:2*nTarget,:,:)
        v = v[0:n_target, :, :] + v[n_target:2 * n_target, :, :]
        r = np.sqrt(v)  # distances: (nTarget, nSensor, nParticles)

        # v = Amp ./ (r.^invPow + d0)
        Amp = float(likeparams["Amp"])
        inv_pow = float(likeparams["invPow"])
        d0 = float(likeparams["d0"])

        v = Amp / (np.power(r, inv_pow) + d0)

        # sum over targets -> (1, nSensor, nParticles)
        v = v.sum(axis=0, keepdims=True)

        # permute to (nSensor, 1, nParticles) then squeeze -> (nSensor, nParticles)
        y = np.squeeze(np.transpose(v, (1, 0, 2)))  # (nSensor, nParticles)

    else:
        # Single particle case, follow MATLAB branch
        # v = bsxfun(@minus,sensorsPos,x);  (2*nTarget, nSensor)
        v = sensors_pos - x  # broadcast over second dimension
        v = v ** 2

        v = v[0:n_target, :] + v[n_target:2 * n_target, :]
        r = np.sqrt(v)  # (nTarget, nSensor)

        Amp = float(likeparams["Amp"])
        inv_pow = float(likeparams["invPow"])
        d0 = float(likeparams["d0"])

        v = Amp / (np.power(r, inv_pow) + d0)
        v = v.sum(axis=0)  # (nSensor,)

        # In MATLAB: y = v'; -> (nSensor,1)
        y = v.reshape(n_sensor, 1)

    return y


# =========================================================
# Acoustic_dh_dxfunc.m
# =========================================================

def acoustic_dh_dxfunc(xp: np.ndarray, likeparams: Dict[str, Any]) -> np.ndarray:
    """
    Python version of Acoustic_dh_dxfunc.m

    Derivative of acoustic_hfunc wrt state xp, with invpow = 1.

    Inputs
    ------
    xp : (dim, nParticles) ndarray
    likeparams : dict
        Must contain:
            'nSensor', 'nTarget', 'sensorsPos', 'Amp', 'd0'

    Returns
    -------
    dhdx :
        If nParticles > 1: (nSensor, dim, nParticles)
        If nParticles == 1: (nSensor, dim)
    """
    xp = np.asarray(xp)
    dim, n_particles = xp.shape

    n_y = int(likeparams["nSensor"])
    n_target = int(likeparams["nTarget"])
    sensors_pos = np.asarray(likeparams["sensorsPos"])  # (2*nTarget, nSensor)

    # Positions
    idx_x = np.arange(0, 4 * n_target, 4)
    idx_y = np.arange(1, 4 * n_target, 4)
    xx = xp[idx_x, :]  # (nTarget, nParticles)
    xy = xp[idx_y, :]  # (nTarget, nParticles)
    x = np.vstack((xx, xy))  # (2*nTarget, nParticles)

    Amp = float(likeparams["Amp"])
    d0 = float(likeparams["d0"])

    if n_particles > 1:
        # dhdx: (nSensor, dim, nParticles)
        dhdx = np.zeros((n_y, dim, n_particles))

        # mv = x - sensorsPos (broadcast)
        # MATLAB: mv = bsxfun(@minus,permute(x,[1 3 2]),likeparams.sensorsPos);
        # permute(x,[1 3 2]) -> (2*nTarget,1,nParticles)
        # sensorsPos         -> (2*nTarget,nSensor)
        # result             -> (2*nTarget,nSensor,nParticles)
        mv = x[:, None, :] - sensors_pos[:, :, None]  # (2*nTarget, nSensor, nParticles)
        v = mv ** 2

        # v(1:nTarget,:,:) = v(1:nTarget,:,:)+v(nTarget+1:2*nTarget,:,:)
        v[0:n_target, :, :] = v[0:n_target, :, :] + v[n_target:2 * n_target, :, :]
        v[n_target:2 * n_target, :, :] = v[0:n_target, :, :]

        # v = sqrt(v);
        r = np.sqrt(v)

        # v = Amp./(((r+d0).^2).*r); then v = -permute(v.*mv,[2 1 3]);
        denom = (r + d0) ** 2 * r
        coef = Amp / denom  # (2*nTarget, nSensor, nParticles)
        v_tmp = -coef * mv  # (2*nTarget, nSensor, nParticles)

        # permute to (nSensor, 2*nTarget, nParticles)
        v_perm = np.transpose(v_tmp, (1, 0, 2))

        # fill dhdx(:, 1:4:nTarget*4, :) and dhdx(:, 2:4:nTarget*4, :)
        # x-derivatives from first nTarget, y-derivatives from next nTarget
        dhdx[:, 0:4 * n_target:4, :] = v_perm[:, 0:n_target, :]
        dhdx[:, 1:4 * n_target:4, :] = v_perm[:, n_target:2 * n_target, :]

        return dhdx

    else:
        # Single particle case
        dhdx = np.zeros((n_y, dim))

        # mv = x - sensorsPos  -> (2*nTarget, nSensor)
        mv = x - sensors_pos
        v = mv ** 2

        v[0:n_target, :] = v[0:n_target, :] + v[n_target:2 * n_target, :]
        v[n_target:2 * n_target, :] = v[0:n_target, :]

        r = np.sqrt(v)
        denom = (r + d0) ** 2 * r
        coef = -Amp / denom  # (2*nTarget, nSensor)

        # v = (coef .* mv)'  -> (nSensor, 2*nTarget)
        v_times_mv = (coef * mv).T  # (nSensor, 2*nTarget)

        # Assign to dhdx(:, 1:4:4*nTarget) / dhdx(:, 2:4:4*nTarget)
        dhdx[:, 0:4 * n_target:4] = v_times_mv[:, 0:n_target]
        dhdx[:, 1:4 * n_target:4] = v_times_mv[:, n_target:2 * n_target]

        return dhdx
