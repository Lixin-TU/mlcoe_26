# pfpf/calculate_errors.py

import numpy as np
from typing import Dict, Any


def calculateErrors(output: Dict[str, Any], ps: Dict[str, Any], alg_name: str) -> None:
    """
    Parameters
    ----------
    output : dict
        Contains at least 'x_est' (dim, T), optionally 'x' (true states).
    ps : dict
        Parameter structure (unused for now).
    alg_name : str
        Name of the algorithm.
    """
    x_est = output.get("x_est", None)
    x_true = output.get("x", None)

    print(f"[calculateErrors] Algorithm: {alg_name}")
    if x_est is not None:
        print(f"  x_est shape: {np.shape(x_est)}")
    if x_true is not None:
        print(f"  x_true shape: {np.shape(x_true)}")
    # TODO: implement real error metrics (RMSE, OSPA, OMAT, etc.)
