from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

from config.HW import *

import xarray as xr

ENGINE = "h5netcdf"
fh = xr.open_dataset("/storage1/HW/paper/5.0_1250_invariants.h5", engine=ENGINE)

Gamma_n = fh["gamma_n"].data
Gamma_c = fh["gamma_c"].data

np.savez("results/Gamma_ref_c1_5.0.npz", Gamma_n=Gamma_n, Gamma_c=Gamma_c)
