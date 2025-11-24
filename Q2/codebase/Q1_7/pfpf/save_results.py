"""
Python translation skeleton of tools/saveResults.m
"""
from typing import Dict, Any, List
from pathlib import Path
from scipy.io import savemat
import numpy as np

def save_results(ps: Dict[str, Any], output: List[Dict[str, Any]], out_dir="Results") -> str:
    
    setup = ps["setup"]
    filename_stub = (
        f"{setup.example_name}_"
        f"{setup.nParticle}particle_"
        f"{setup.dimState_all}dimension_"
        f"{setup.nTrack}tracks_"
        f"{setup.nAlg_per_track}runs"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # algorithm name string like in MATLAB
    alg_string = ""
    for alg in setup.algs_executed:
        alg_string += f"_{alg}"

    if setup.example_name == "Acoustic":
        filename = filename_stub + alg_string + ".mat"
    elif setup.example_name == "Septier16":
        # You can mirror the detailed naming convention from saveResults.m
        raise NotImplementedError("Septier16 filename pattern not implemented yet.")
    else:
        filename = filename_stub + alg_string + ".mat"

    full_path = out_dir / filename

    # Convert ps and output into savemat-friendly structures.
    # Minimal version: store ps as a nested dict, and output as a list of dicts.
    mat_dict = {
        "ps": {
            "setup": setup.__dict__ if hasattr(setup, "__dict__") else dict(setup),
            # add other sub-structs here (likeparams, dynparams, initparams, ...)
        },
        "output": output,
    }

    savemat(full_path, mat_dict)
    return str(full_path)
