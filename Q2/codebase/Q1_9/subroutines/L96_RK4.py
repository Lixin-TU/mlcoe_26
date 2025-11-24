import numpy as np

def L96_RK4(X_in, dt, F):
    """
    integrate for L96 model using RK4
    X_in is the input, dt is the time resolution
    X_out is the output
    """
    
    dim, np_particles = X_in.shape
    
    # Define the L96 derivative function to avoid code repetition
    def l96_deriv(X):
        # In Python numpy, we can use roll to shift indices efficiently
        # x(i-1) corresponds to roll(X, 1)
        # x(i-2) corresponds to roll(X, 2)
        # x(i+1) corresponds to roll(X, -1)
        
        X_p1 = np.roll(X, -1, axis=0) # x(i+1)
        X_n1 = np.roll(X, 1, axis=0)  # x(i-1)
        X_n2 = np.roll(X, 2, axis=0)  # x(i-2)
        
        dX = (X_p1 - X_n2) * X_n1 - X + F
        return dX

    k1 = l96_deriv(X_in)
    
    tmp_b = X_in + 0.5 * k1 * dt
    k2 = l96_deriv(tmp_b)
    
    tmp_b = X_in + 0.5 * k2 * dt
    k3 = l96_deriv(tmp_b)
    
    tmp_b = X_in + k3 * dt
    k4 = l96_deriv(tmp_b)
    
    X_out = X_in + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return X_out