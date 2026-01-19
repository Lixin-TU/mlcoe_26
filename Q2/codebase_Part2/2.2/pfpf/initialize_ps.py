import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pfpf.acoustic_model import (
    acoustic_propagate,
    acoustic_hfunc,
    acoustic_dh_dxfunc,
)
from pfpf.particle_flow import particle_flow
from .acoustic_init import acoustic_gauss_init
from .likelihoods import gaussian_llh



# ========== dataclasses for clean structure ==========
@dataclass
class Setup:
    example_name: str = "Acoustic"
    doplot: bool = True
    parallel_run: bool = False
    algs_executed: List[str] = field(default_factory=list)
    lambda_type: str = "exponential"
    nLambda: int = 29
    PFPF: bool = True
    kflag: str = "EKF1"
    regularize_resample: bool = False
    fontSize: int = 20
    Resampling: bool = True
    Redraw: bool = True
    maxilikeSAP: int = 200
    maxilikemode: str = "a"
    use_cluster: bool = False

    # cluster params
    weight_euclidean: float = 0.25
    nParticleCluster: int = 100
    doplot_cluster: bool = False

    Neff_thresh_ratio: float = 0.5

    # to be filled during initialization
    lambda_range: np.ndarray = None
    nTrack: int = None
    nAlg_per_track: int = None
    nParticle: int = None
    T: int = None
    nTarget: int = None
    dimState_per_target: int = None
    dimState: int = None
    dimState_all: int = None
    ospa_c: float = None
    ospa_p: float = None
    random_seeds: np.ndarray = None
    plotfcn: Any = None


# ========== helper: exponential lambda ==========
def generate_exponential_lambda(n):
    lam = np.linspace(0, 1, n)
    return lam**2  


# ========== main initializer ==========
def initialize_ps(algs_executed: List[str]) -> Dict[str, Any]:
    ps: Dict[str, Any] = {}
    setup = Setup()
    setup.algs_executed = list(algs_executed)
    ps["setup"] = setup

    # ==== lambda range ====
    if setup.lambda_type == "uniform":
        setup.lambda_range = np.linspace(0, 1, setup.nLambda)
    else:
        setup.lambda_range = generate_exponential_lambda(setup.nLambda)

    # ==== dataset-specific initialization ====
    if setup.example_name == "Acoustic":
        ps = acoustic_example_initialization(ps)

    # ==== number of trials ====
    setup.nTrial = setup.nTrack * setup.nAlg_per_track

    # ==== random seeds ====
    setup.random_seeds = np.random.choice(
        int(1e5 * setup.nTrial),
        size=setup.nTrial,
        replace=False,
    )

    # ==== global state dimension ====
    if setup.nTarget is not None:
        setup.dimState_all = setup.nTarget * setup.dimState_per_target
        setup.dimState = setup.dimState_all
    else:
        setup.dimState_all = setup.dimState

    # ==== SmHMC params (placeholder) ====
    ps["SmHMC_model_params"] = {"dummy": True}  # 以后再补 generateSmHMCModelParam()

    return ps


# =====================================================================
# ========== Acoustic Example Initialization (translation) ==========
# =====================================================================

def acoustic_example_initialization(ps: Dict[str, Any]) -> Dict[str, Any]:
    setup = ps["setup"]

    # ===== basic parameters =====
    setup.nTrack = 1
    setup.nAlg_per_track = 5
    setup.nParticle = 500

    setup.T = 40
    setup.nTarget = 4
    setup.dimState_per_target = 4

    setup.ospa_c = 40
    setup.ospa_p = 1

    nTarget = setup.nTarget

    # ===== load sensorsXY (must be placed in the same folder) =====
    import scipy.io
    from pathlib import Path
    
    # Try finding sensorsXY.mat in likely locations
    possible_paths = [
        "sensorsXY.mat",
        "scripts/sensorsXY.mat",
        "../scripts/sensorsXY.mat",
        Path(__file__).resolve().parent.parent / "scripts/sensorsXY.mat"
    ]
    
    sensorsXY = None
    for p in possible_paths:
        try:
            sensorsXY = scipy.io.loadmat(str(p))["sensorsXY"]
            break
        except FileNotFoundError:
            continue
            
    if sensorsXY is None:
        raise FileNotFoundError("Could not find sensorsXY.mat in expected locations.")

    simAreaSize = 40
    sensorsPos = simAreaSize / 40 * sensorsXY
    nSensor = sensorsPos.shape[0]

    sensorsPos = sensorsPos.T
    sensorsPos = np.kron(sensorsPos, np.ones((nTarget, 1)))

    # ===== init state for 4 targets =====
    x0 = np.array([
        12, 6, 0.001, 0.001,
        32, 32, -0.001, -0.005,
        20, 13, -0.1,  0.01,
        15, 35, 0.002, 0.002
    ]).reshape(-1, 1)

    survRegion = np.array([0, 0, 40, 40])
    trackBounds = np.array([-10, -10, 50, 50])

    # ===== motion model =====
    Phi_single = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Gamma = np.array([
        [1/3, 0,   0.5,  0],
        [0,   1/3, 0,    0.5],
        [0.5, 0,   1,    0],
        [0,   0.5, 0,    1]
    ])

    gammavar_real = 0.05
    Qii_real = gammavar_real * Gamma

    Qii = np.array([
        [3,   0,   0.1, 0],
        [0,   3,   0,   0.1],
        [0.1, 0,   0.03, 0],
        [0,   0.1, 0,   0.03]
    ])

    Q_ii_corr = 0.1 * np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0.01, 0],
                                [0, 0, 0, 0.01]])

    # replicate for all targets
    Phi = Phi_single.copy()
    Q = Qii.copy()
    Q_real = Qii_real.copy()
    Q_corr = Q_ii_corr.copy()
    for _ in range(nTarget - 1):
        Phi = scipy.linalg.block_diag(Phi, Phi_single)
        Q = scipy.linalg.block_diag(Q, Qii)
        Q_real = scipy.linalg.block_diag(Q_real, Qii_real)
        Q_corr = scipy.linalg.block_diag(Q_corr, Q_ii_corr)

    # ===== ps.initparams =====
    ps["initparams"] = {
        "x0": x0,
        "sigma0": np.tile(10 * np.array([1, 1, 0.1, 0.1]).reshape(-1, 1), (nTarget, 1)),
        "nTarget": nTarget,
        "survRegion": survRegion,
        "simAreaSize": simAreaSize,
    }

    # ===== propparams =====
    ps["propparams"] = {
        "Phi": Phi,
        "Q": Q,
        "Q_correction": Q_corr,
        "Q_regularized": Q_corr,
        "nTarget": nTarget,
        "propagatefcn": acoustic_propagate, 
        "dimState_per_target": setup.dimState_per_target,
    }

    ps["propparams_real"] = {
        "Phi": Phi,
        "Q": Q_real,
        "nTarget": nTarget
    }

    # ===== measurement params =====
    measvar_real = 0.01
    measvar = measvar_real

    ps["likeparams"] = {
        "sensorsPos": sensorsPos,
        "Amp": 10,
        "d0": 0.1,
        "invPow": 1,
        "measvar": measvar,
        "measvar_real": measvar_real,
        "noise": "Gaussian",
        "simAreaSize": simAreaSize,
        "nTarget": nTarget,
        "nSensor": nSensor,
        "dimMeasurement_per_target": nSensor,
        "dimMeasurement_all": nSensor,
        "survRegion": survRegion,
        "trackBounds": trackBounds,
    }

    ps["likeparams"]["R_real"] = measvar_real * np.eye(nSensor)
    ps["likeparams"]["R"] = measvar * np.eye(nSensor)
    ps["likeparams"]["R_inv"] = np.linalg.inv(ps["likeparams"]["R"])

    setup.plotfcn = "AcousticParticlePlot"

    ps["likeparams"]["llh"] = gaussian_llh
    ps["likeparams"]["h_func"] = acoustic_hfunc
    ps["likeparams"]["dh_dx_func"] = acoustic_dh_dxfunc
    ps["likeparams"]["dH_dx_func"] = "Acoustic_dH_dx"
    ps["initparams"]["initfcn"] = acoustic_gauss_init

    return ps
