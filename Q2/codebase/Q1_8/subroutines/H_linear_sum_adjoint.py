import numpy as np

def H_linear_sum_adjoint(X, w):
    """
    subroutine for the adjoint of the observation operator (analytical
    solution) for "one observation"
    input:
    X : ensmemble in state space (size: [# state variable in inner domain * # of ens member])
    w : the weights for each state variable in inner domain
    output:
    dHdx: adjoint (size: [# state variable in inner domain * # of ens member])
    2022/02/25
    """
    dim_inner, np_particles = X.shape
    if len(w) != dim_inner:
        raise ValueError('length of the weights should be equal to the inner domain size')

    dHdx = np.tile(np.reshape(w, (dim_inner, 1)), (1, np_particles))
    
    return dHdx