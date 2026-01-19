from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

# Defaults for Acoustic example, used if not found in ps.setup
OSPA_C_DEFAULT = 40.0
OSPA_P_DEFAULT = 1.0
DIM_STATE_PER_TARGET_DEFAULT = 4  # [x, y, vx, vy]


def ospa_dist(X, Y, c, p):
    """
    X, Y : 2 x n matrices (each column is a target position)
    c    : cut-off parameter
    p    : p-parameter

    Returns
    -------
    dist : scalar OSPA distance
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # Ensure shape is (dim, n)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim != 2:
        Y = Y.reshape(Y.shape[0], -1)

    # Handle empty sets
    if X.size == 0 and Y.size == 0:
        return 0.0
    if X.size == 0 or Y.size == 0:
        return float(c)

    n = X.shape[1]
    m = Y.shape[1]

    # Cost matrix D (n x m): ||X_i - Y_j||, clipped by c and raised to p
    # Vectorized version similar to MATLAB
    XX = np.repeat(X, m, axis=1)                          # (dim, n*m)
    YY = np.tile(Y, (1, n))                               # (dim, n*m)
    D = np.sqrt(np.sum((XX - YY) ** 2, axis=0))           # (n*m,)
    D = D.reshape(n, m)
    D = np.minimum(c, D) ** p

    # Hungarian algorithm to get optimal assignment
    row_ind, col_ind = linear_sum_assignment(D)
    cost = D[row_ind, col_ind].sum()

    dist = (1.0 / max(m, n) * (c**p * abs(m - n) + cost)) ** (1.0 / p)
    return float(dist)


# ---------- Helpers to read MATLAB-style ps + output ----------

def load_pipeline_results(mat_path):
    """
    Load results from the MAT file saved by run_pipeline.py.

    Returns
    -------
    meta : dict with keys
        - nTrial
        - T
        - algs
        - output_structs
        - dim
        - ps  (mat_struct or None)
    """
    mat_path = Path(mat_path)
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    ps_struct = mat.get("ps", None)

    if "output" not in mat:
        raise KeyError("Field 'output' not found in MAT file.")

    output = mat["output"]

    # After squeeze, output is typically a 1D array of mat_struct
    if not isinstance(output, np.ndarray):
        output_structs = np.array([output])
    else:
        output_structs = output
        if output_structs.ndim == 0:
            output_structs = np.array([output_structs])

    nTrial = output_structs.shape[0]

    # Use first trial to infer alg names and T
    first_trial = output_structs[0]
    algs = list(first_trial._fieldnames)

    first_alg_name = algs[0]
    first_alg_struct = getattr(first_trial, first_alg_name)

    x = np.asarray(first_alg_struct.x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    dim, T = x.shape

    return {
        "nTrial": nTrial,
        "T": T,
        "algs": algs,
        "output_structs": output_structs,
        "dim": dim,
        "ps": ps_struct,
    }


# ---------- Main plotting: OSPA/OMAT-based version of plotErrors.m ----------

def plot_errors_python_mat(mat_path, out_dir=None, show=False):
    """
    compute per-step OSPA distance over time, averaged over trials
    compute a scalar per-trial mean OSPA error
    compute mean ESS per trial (if Neff exists)
    """
    mat_path = Path(mat_path)
    meta = load_pipeline_results(mat_path)

    nTrial = meta["nTrial"]
    T = meta["T"]
    algs = meta["algs"]
    output_structs = meta["output_structs"]
    dim = meta["dim"]
    ps = meta["ps"]

    # Try to read parameters from ps; fall back to Acoustic defaults
    c = OSPA_C_DEFAULT
    p = OSPA_P_DEFAULT
    dim_state_per_target = DIM_STATE_PER_TARGET_DEFAULT

    if ps is not None and hasattr(ps, "setup"):
        setup = ps.setup
        if hasattr(setup, "ospa_c"):
            c_val = np.asarray(setup.ospa_c).flatten()
            if c_val.size > 0:
                c = float(c_val[0])
        if hasattr(setup, "ospa_p"):
            p_val = np.asarray(setup.ospa_p).flatten()
            if p_val.size > 0:
                p = float(p_val[0])
        if hasattr(setup, "dimState_per_target"):
            d_val = np.asarray(setup.dimState_per_target).flatten()
            if d_val.size > 0:
                dim_state_per_target = int(d_val[0])

    # Infer number of targets from state dimension
    if dim % dim_state_per_target != 0:
        raise ValueError(
            f"State dimension {dim} is not divisible by dimState_per_target={dim_state_per_target}."
        )
    nTarget = dim // dim_state_per_target

    nAlg = len(algs)

    # Containers
    error_per_trial_per_alg = np.zeros((nTrial, nAlg))
    ESS_per_trial_per_alg = np.zeros((nTrial, nAlg))
    execution_time_per_alg = np.zeros((nAlg,))

    per_step_error_mean = {}
    per_step_neff_mean = {}

    t_axis = np.arange(1, T + 1)

    # Loop over algorithms
    for j, alg in enumerate(algs):
        error_steps_all = np.zeros((T, nTrial))
        neff_steps_all = np.zeros((T, nTrial))
        has_neff = False
        exec_times = []

        for i in range(nTrial):
            trial_struct = output_structs[i]
            alg_struct = getattr(trial_struct, alg)

            # True state and estimate
            x = np.asarray(alg_struct.x)
            x_est = np.asarray(alg_struct.x_est)

            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if x_est.ndim == 1:
                x_est = x_est.reshape(-1, 1)

            if x.shape != x_est.shape:
                # Try transpose both if shapes are (T, dim)
                x = x.T
                x_est = x_est.T

            # Now x, x_est : (dim, T)
            if x.shape[0] != dim:
                # Try transposing once more if needed
                x = x.T
                x_est = x_est.T

            if x.shape != x_est.shape:
                raise ValueError(f"Shape mismatch between x and x_est for alg {alg}, trial {i}.")

            dim_check, T_check = x.shape
            if dim_check != dim or T_check != T:
                raise ValueError(f"Unexpected shape for x: {x.shape}, expected ({dim},{T}).")

            # Compute OSPA per time step
            errors_t = np.zeros(T)
            for t in range(T):
                x_t = x[:, t]
                x_est_t = x_est[:, t]

                # Reshape (dim,) -> (dim_state_per_target, nTarget) using column-major (Fortran) order
                x_true_mat = x_t.reshape(dim_state_per_target, nTarget, order="F")
                x_est_mat = x_est_t.reshape(dim_state_per_target, nTarget, order="F")

                true_tracks = x_true_mat[0:2, :]      # positions only
                est_tracks = x_est_mat[0:2, :]

                errors_t[t] = ospa_dist(est_tracks, true_tracks, c, p)

            error_steps_all[:, i] = errors_t
            error_per_trial_per_alg[i, j] = float(errors_t.mean())

            # ESS if available
            if hasattr(alg_struct, "Neff"):
                neff = np.asarray(alg_struct.Neff).reshape(-1)
                if neff.shape[0] == T:
                    neff_steps_all[:, i] = neff
                    has_neff = True

            # Execution time
            if hasattr(alg_struct, "execution_time"):
                exec_times.append(float(np.squeeze(alg_struct.execution_time)))

        # Average over trials
        err_mean = np.mean(error_steps_all, axis=1)   # (T,)
        per_step_error_mean[alg] = err_mean

        if has_neff:
            neff_mean = np.mean(neff_steps_all, axis=1)
            per_step_neff_mean[alg] = neff_mean
            ESS_per_trial_per_alg[:, j] = np.mean(neff_steps_all, axis=0)
        else:
            ESS_per_trial_per_alg[:, j] = 0.0

        if exec_times:
            execution_time_per_alg[j] = float(np.mean(exec_times))
        else:
            execution_time_per_alg[j] = np.nan

    # ----- Combined Figure-----
    line_styles = {
    "EDH": "^-",          
    "LEDH": "d-",         
    "PFPF_EDH": "o-",     
    "PFPF_LEDH": "s-",
    "PFPF_LEDH_Optimal": "x-",
    }

    algorithm_colors = {
    "EDH": "#1f77b4",         
    "LEDH": "#ff7f0e",        
    "PFPF_EDH": "#2ca02c",   
    "PFPF_LEDH": "#d62728",
    "PFPF_LEDH_Optimal": "#9467bd",
}


    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ------------------------------------------------------------------
    # Subplot 1: per-step OMAT error
    # ------------------------------------------------------------------
    ax1 = axes[0]
    for alg in algs:
        style = line_styles.get(alg, '-')   
        color = algorithm_colors.get(alg, "black")
        ax1.plot(
            t_axis,
            per_step_error_mean[alg],
            style,
            color=color,
            markevery=3,
            markerfacecolor='none',
            markersize=7,
            linewidth=2,
            label=alg
        )
    ax1.set_xlabel("time step")
    ax1.set_ylabel("average OMAT error (m)")
    ax1.legend()

    # ------------------------------------------------------------------
    # Subplot 2: per-step ESS
    # ------------------------------------------------------------------
    ax2 = axes[1]
    any_neff = False
    for alg, neff_mean in per_step_neff_mean.items():
        style = line_styles.get(alg, '-')   
        color = algorithm_colors.get(alg, "black")
        any_neff = True
        ax2.plot(
            t_axis,
            neff_mean,
            style,
            color=color,
            markevery=3,
            markerfacecolor='none',
            markersize=7,
            linewidth=2,
            label=alg
        )

    if any_neff:
        ax2.set_xlabel("time step")
        ax2.set_ylabel("average ESS")
        ax2.legend()


    # Subplot 3: boxplot of per-trial OMAT error
    ax3 = axes[2]
    bp_data = [error_per_trial_per_alg[:, j] for j in range(nAlg)]
    ax3.boxplot(bp_data, tick_labels=algs)
    # ax3.set_xlabel("Algorithm")
    ax3.set_ylabel("OMAT error")
    # ax3.set_title("Filtering error per algorithm (OMAT boxplot over trials)")
    # ax3.grid(True, axis="y")
    
    plt.tight_layout()

    # Print summary
    print("------------------------------------------------------------")
    print("Summary of OMAT errors and ESS:")
    for j, alg in enumerate(algs):
        err_mean_trials = error_per_trial_per_alg[:, j].mean()
        ess_mean_trials = ESS_per_trial_per_alg[:, j].mean()
        exec_time = execution_time_per_alg[j]
        print(f"  {alg}:")
        print(f"    mean OMAT over time  = {err_mean_trials:.4f}")
        print(f"    mean ESS over time   = {ess_mean_trials:.2f}")
        print(f"    avg execution time (s) = {exec_time:.4f}")
    print("------------------------------------------------------------")

    # Save figure
    out_paths = {}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        p1 = out_dir / "combined_plots.png"
        fig.savefig(p1, dpi=200, bbox_inches="tight")
        out_paths["combined"] = str(p1)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "algs": algs,
        "error_per_trial_per_alg": error_per_trial_per_alg,
        "ESS_per_trial_per_alg": ESS_per_trial_per_alg,
        "execution_time_per_alg": execution_time_per_alg,
        "out_paths": out_paths,
        "T": T,
        "nTrial": nTrial,
    }


if __name__ == "__main__":
    # Use the multi-trial MAT file saved by run_pipeline.py
    mat_path = "./Results/tracking_results_acoustic.mat"
    res = plot_errors_python_mat(mat_path, out_dir="Plots", show=True)
    print("Algorithms:", res["algs"])
