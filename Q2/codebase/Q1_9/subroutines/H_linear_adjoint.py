import numpy as np

def H_linear_adjoint(X):
    """
    subroutine for the adjoint of the observation operator (analytical
    solution) for "one observation"
    input:
    X : ensmemble in state space (size: [# state variable in inner domain * # of ens member])
    output:
    dHdx: adjoint (size: [# state variable in inner domain * # of ens member])
    """
    dim_inner, np_particles = X.shape
    dHdx = np.ones((dim_inner, np_particles))
    return dHdx