import numpy as np
from subroutines.inv_SVD import inv_SVD

def adjoint_pseudoinverse(X, Hx, cond_num):
    """
    subroutine for estimating the adjoint of the observation operator (by assuming linear H)
    input:
    X : ensmemble in state space (size: [# state variable in inner domain * # of ens member])
    Hx: ensemble in the observation space (size: [# of ens member])
    cond_num: condition number used for SVD 
    output:
    dHdx: adjoint (size: [# state variable in inner domain * # of ens member])
    """
    dim_inner, np_particles = X.shape
    
    # pseudo_inv_X = inv_SVD(X',cond_num,1);
    pseudo_inv_X = inv_SVD(X.T, cond_num, 1)
    
    # HT = pseudo_inv_X*reshape(Hx, [np 1]);
    HT = pseudo_inv_X @ Hx.reshape(np_particles, 1)
    
    # dHdx = repmat(HT, [1 np]); 
    dHdx = np.tile(HT, (1, np_particles))
    
    return dHdx