import numpy as np

def H_linear_sum(X, w):
    """
    observation operator for linear weighted average obs
    input:
    X : ensmemble in state space (size: [# state variable in inner domain * # of ens member])
    w : the weights for each state variable in inner domain
    output:
    Hx: ensemble in the observation space (size: [# of ens member])
    2022/02/25
    """
    dim_inner, np_particles = X.shape
    
    if len(w) != dim_inner:
        raise ValueError('length of the weights should be equal to the inner domain size')
    
    # Reshape w to (dim_inner, 1) and tile
    W = np.tile(np.reshape(w, (dim_inner, 1)), (1, np_particles))
    
    # Sum along the first axis (rows), equivalent to sum(..., 1) in MATLAB for this shape
    Hx = np.sum(W * X, axis=0) 
    
    return Hx