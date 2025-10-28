from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

import xarray as xr

from config.HW import *

if __name__ == "__main__":
    data = np.load(POD_file)
    Vr = data["Vr"][:, :8]

    np.save("results/Vr.npy", Vr)
