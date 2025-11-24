import numpy as np
from scipy.linalg import svd

def inv_SVD(A, cond_num, option):
    """
    compute the inverse of a matrix A using the given condition number
    option = 1: calculate the inverse of A
    option = 2: calculate the square root of A
    """
    
    U, s, Vh = svd(A)
    # V = Vh.T
    
    sig_val = s
    sig_inv = np.zeros((len(s), len(s)))
    
    if option == 1:
        for i in range(len(sig_val)):
            if sig_val[i] < (10**cond_num) * sig_val[0]:
                sig_inv[i, i] = 0 # zero out too small singular value
            else:
                sig_inv[i, i] = 1.0 / sig_val[i]
        
        # A_inv = V * sig_inv * U'
        # In Python terms: Vh.T @ sig_inv @ U.T
        A_inv = Vh.T @ sig_inv @ U.T
        
    elif option == 2:
        for i in range(len(sig_val)):
            if sig_val[i] < (10**cond_num) * sig_val[0]:
                sig_inv[i, i] = 0
            else:
                sig_inv[i, i] = np.sqrt(sig_val[i])
                
        A_inv = U @ sig_inv @ Vh # Wait, MATLAB says u*sig_inv*v'. 

        A_inv = U @ sig_inv @ Vh # Keep strict translation of the formula used in MATLAB lines
        
    return A_inv