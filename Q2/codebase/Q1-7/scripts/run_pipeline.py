from pathlib import Path
import sys
import copy
import numpy as np
import h5py
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "Results"
SIMDATA_PATH = RESULTS_DIR / "Acoustic_SimData.mat"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pfpf import initialize_ps, run_one_trial


def save_results_matlab_style(ps, output, out_path):
    """
    Save results 
    """

    setup = ps["setup"]
    nTrial = setup.nTrial
    algs = setup.algs_executed

    output_cell = [[None] * nTrial]  # 1 x nTrial cell

    for t in range(nTrial):
        trial_dict = output[t]  # e.g. {"PFPF_LEDH": {...}, ...}
        matlab_struct = {}

        for alg in algs:
            if alg in trial_dict:
                matlab_struct[alg] = trial_dict[alg]

        output_cell[0][t] = matlab_struct

    setup_dict = {
        "T":             setup.T,
        "nTrial":        setup.nTrial,
        "nTrack":        setup.nTrack,
        "nAlg_per_track":setup.nAlg_per_track,
        "nParticle":     setup.nParticle,
        "example_name":  setup.example_name,
    }

    for key in ["ospa_c", "ospa_p", "nTarget", "dimState_per_target"]:
        if hasattr(setup, key):
            val = getattr(setup, key)
            if val is not None:
                setup_dict[key] = val

    setup_dict["algs_executed"] = np.array(algs, dtype=object)
    ps_save = {"setup": setup_dict}

    savemat(out_path, {"ps": ps_save, "output": output_cell})
    return out_path



def run(algs_executed):

    # Step 1 — build ps
    ps_initial = initialize_ps(algs_executed)

    ps_initial["setup"].doplot = False
    ps_initial["setup"].plotfcn = None


    with h5py.File(SIMDATA_PATH, "r") as f:
        x_all = np.array(f["x_all"])   # (16, 40) in MATLAB, but (40,16) here
        y_all = np.array(f["y_all"])   # (25, 40) in MATLAB, but (40,25) here

    x_all = x_all.T       
    y_all = y_all.T        

    ps_initial["x_all"] = [x_all]   # single track
    ps_initial["y_all"] = [y_all]


    nTrial = ps_initial["setup"].nTrial

    # Step 3 — run trials
    output = [None] * nTrial

    for trial_ix in range(nTrial):
        ps_trial = copy.deepcopy(ps_initial)
        ps_trial["trial_ix"] = trial_ix + 1
        output[trial_ix] = run_one_trial(ps_trial)

    # Step 4 — save
    out_path = RESULTS_DIR / "tracking_results_acoustic.mat"
    save_results_matlab_style(ps_initial, output, out_path)

    print("\nSaved results to:", out_path)
    return str(out_path)


if __name__ == "__main__":
    algs = ["EDH", "LEDH", "PFPF_EDH", "PFPF_LEDH"]
    run(algs)
