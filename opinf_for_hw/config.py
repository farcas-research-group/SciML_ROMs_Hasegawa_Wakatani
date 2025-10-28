from pathlib import Path

import numpy as np

# switch to choose predictions beyond training data or for multiple initial conditions
MULTIPLE_IC = False

R = 138  # or 60

ENGINE = "h5netcdf"

DW = 2
DT = 2.5e-2
L = 2 * np.pi / 0.15


NX = 512
NY = 512
N_CELLS = NX * NY

T_INIT = 20000
TRAINING_START = 0
TRAINING_END = 10000
TRAINING_START_TIME = 500
TRAINING_END_TIME = 749.9999

TRAINING_SIZE = TRAINING_END - TRAINING_START

SVD_SAVE = 1000

N_STEPS = 20001 - TRAINING_START

R_ALL = [60, 138]

if MULTIPLE_IC:
    DATA_SUBDIR = "multiple_ic"
    N_STEPS = 10000 - TRAINING_START

    time_steps_rec = [0, 4000, 7000, 10000, 11000, 14000, 17000, 20000]
    time_steps_rec = [0, 4000, 7000, 10000]

else:
    DATA_SUBDIR = "beyond_training"
    N_STEPS = 20001 - TRAINING_START

    time_steps_rec = [
        999,
        1999,
        3999,
        5999,
        7999,
        9999,
        10000,
        11999,
        12999,
        13999,
        14999,
        15999,
        16999,
        17999,
        18999,
        20000,
    ]

times_rec = [500 + DT * x for x in time_steps_rec]

BASIS_SUBDIR = Path("projected_basis")
MEAN_FILE = BASIS_SUBDIR / "mean.npy"
SCL_FILE = BASIS_SUBDIR / "scl.npy"
POD_FILE = BASIS_SUBDIR / "POD.npz"
XHAT_FILE = BASIS_SUBDIR / "X_hat.npy"

REFERENCE_SUBDIR = Path("reference")
