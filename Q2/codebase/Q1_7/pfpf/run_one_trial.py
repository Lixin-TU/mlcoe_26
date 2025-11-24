# pfpf/run_one_trial.py

import numpy as np
import copy
from typing import Dict, Any

from .pfpf_filter import PFPF
from .dh_exact_flow_filter import DH_ExactFlow_Filter
from .calculate_errors import calculateErrors


def run_one_trial(ps: Dict[str, Any]) -> Dict[str, Any]:

    tracking_output: Dict[str, Any] = {}

    trial_ix = int(ps["trial_ix"])

    # Only one track exists → always use index 0
    ps["x"] = ps["x_all"][0]        # shape (16, 40)
    y = ps["y_all"][0]              # shape (25, 40)

    algs_executed = ps["setup"].algs_executed
    base_seed = int(ps["setup"].random_seeds[trial_ix - 1])

    def _reset_rng():
        np.random.seed(base_seed)

    # ------------------------------------------------
    if "PFPF_LEDH" in algs_executed:
        _reset_rng()
        ps2 = copy.deepcopy(ps)
        ps2["setup"].PFPF = True
        ps2["setup"].pf_type = "LEDH"
        tracking_output["PFPF_LEDH"] = PFPF(ps2, y)

    if "PFPF_EDH" in algs_executed:
        _reset_rng()
        ps2 = copy.deepcopy(ps)
        ps2["setup"].PFPF = True
        ps2["setup"].pf_type = "EDH"
        tracking_output["PFPF_EDH"] = PFPF(ps2, y)

    if "LEDH" in algs_executed:
        _reset_rng()
        ps2 = copy.deepcopy(ps)
        ps2["setup"].PFPF = False
        ps2["setup"].pf_type = "LEDH"
        tracking_output["LEDH"] = DH_ExactFlow_Filter(ps2, y)

    if "EDH" in algs_executed:
        _reset_rng()
        ps2 = copy.deepcopy(ps)
        ps2["setup"].PFPF = False
        ps2["setup"].pf_type = "EDH"
        tracking_output["EDH"] = DH_ExactFlow_Filter(ps2, y)

    # ------------------------------------------------
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print(f"Displaying results for selected algorithms, Trial {trial_ix}:")
    for alg in tracking_output:
        calculateErrors(tracking_output[alg], ps, alg)
    print("------------------------------------------------------------")
    print("------------------------------------------------------------")

    return tracking_output
