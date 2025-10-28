from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *

import xarray as xr

from config.HW import *

if __name__ == "__main__":
    data = np.load(POD_file)
    Vr = data["Vr"][:, :8]

    np.save("results/Vr.npy", Vr)
