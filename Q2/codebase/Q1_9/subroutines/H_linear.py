import numpy as np

def H_linear(X):
    """
    observation operator for linear obs
    input:
    X : ensmemble in state space (size: [# state variable in inner domain * # of ens member])
    output:
    Hx: ensemble in the observation space (size: [# of ens member])
    2022/02/25
    """
    # [dim_inner, np] = size(X)
    Hx = X
    return Hx