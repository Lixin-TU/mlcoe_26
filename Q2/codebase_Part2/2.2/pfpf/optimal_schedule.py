import numpy as np
from scipy.integrate import solve_bvp
from numpy.linalg import inv, pinv

def get_optimal_schedule(P_prior, J, R, n_steps, mu_weight=0.2):
    """
    Solves the Boundary Value Problem (BVP) from Dai(22) to find the optimal 
    lambda (beta) schedule.

    Parameters
    ----------
    P_prior : (dim, dim) array
        Prior covariance matrix.
    J : (m, dim) array
        Jacobian of measurement function at the prior mean.
    R : (m, m) array
        Measurement noise covariance.
    n_steps : int
        Number of steps to discretize the schedule (returns n_steps + 1 points).
    mu_weight : float
        Weight parameter for the scalar metric (default 0.2 from q21).

    Returns
    -------
    lambdas : (n_steps + 1,) array
        Optimal lambda schedule from 0 to 1.
    """
    dim = P_prior.shape[0]

    # M0 = inv(P_prior)
    # Using pseudoinverse for stability if P_prior is near singular, 
    # though it should be PD.
    try:
        M0 = inv(P_prior)
    except:
        M0 = pinv(P_prior)

    # Mh = J.T @ inv(R) @ J (Fisher Information / Hessian approximation)
    try:
        R_inv = inv(R)
    except:
        R_inv = pinv(R)
    
    Mh = J.T @ R_inv @ J

    # Define ODE system
    # y[0] = beta
    # y[1] = u (dbeta/dt)
    
    def ode(tau, Y):
        beta = Y[0]
        u = Y[1]
        d2beta = np.zeros_like(beta)
        
        # We process vectorized tau if needed, but usually solve_bvp passes arrays
        n_points = beta.shape[0]
        
        for i in range(n_points):
            b = np.clip(beta[i], 0, 1)
            
            M = M0 + b * Mh + 1e-9 * np.eye(M0.shape[0]) # Regularization for stability
            try:
                M_inv = inv(M)
            except:
                M_inv = pinv(M)
            
            # Eq (28) from Dai(22)
            # Minimize trace(M^-1) ? No, the paper minimizes a specific metric.
            # q21 implementation:
            # term1 = np.trace(Mh) * np.trace(M_inv)
            # term2 = np.trace(M) * np.trace(M_inv @ M_inv @ Mh)
            # d2beta = -mu_weight * (term1 + term2)
            # Note: The sign and form come from the variation of the cost functional.
            
            t_Mh = np.trace(Mh)
            t_Minv = np.trace(M_inv)
            t_M = np.trace(M)
            t_Minv2_Mh = np.trace(M_inv @ M_inv @ Mh)
            
            d2beta[i] = -mu_weight * (t_Mh * t_Minv + t_M * t_Minv2_Mh)
            
        return np.vstack((u, d2beta))

    def bc(ya, yb):
        # beta(0) = 0, beta(1) = 1
        return np.array([ya[0], yb[0] - 1])

    # Initial guess
    x_mesh = np.linspace(0, 1, 20)
    y_guess = np.zeros((2, len(x_mesh)))
    y_guess[0] = x_mesh 
    y_guess[1] = 1.0    
    
    # Solve BVP
    res = solve_bvp(ode, bc, x_mesh, y_guess, tol=1e-3, max_nodes=1000)

    # If successful, interpolate
    desired_mesh = np.linspace(0, 1, n_steps + 1)
    if res.success:
        beta_vals = res.sol(desired_mesh)[0]
        # Ensure strict 0 and 1
        beta_vals[0] = 0.0
        beta_vals[-1] = 1.0
        # Ensure monotonicity
        beta_vals = np.sort(beta_vals)
    else:
        print("Warning: Optimal schedule BVP failed. Fallback to linear.")
        beta_vals = desired_mesh

    return beta_vals
