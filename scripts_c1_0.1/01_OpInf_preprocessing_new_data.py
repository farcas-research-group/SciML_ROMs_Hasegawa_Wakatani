from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

import xarray as xr

from config.HW import *

if __name__ == "__main__":
    print("\033[1m Reading snapshots...\033[0m")

    ENGINE = "h5netcdf"

    fh = xr.open_dataset("/storage1/HW/paper/0.10_300_training.h5", engine=ENGINE)
    Q_train = (
        xr.concat(
            [
                fh["density"].expand_dims(dim={"field": ["density"]}, axis=1),
                fh["potential"].expand_dims(dim={"field": ["potential"]}, axis=1),
            ],
            dim="field",
        )
        .stack(n=("field", "y", "x"))
        .transpose("n", "time")
        .data
    )
    print(Q_train)

    # load POD data
    print("\033[1m Compute POD basis...\033[0m")
    U, S, _ = np.linalg.svd(Q_train, full_matrices=False)
    np.savez(POD_file, S=S, Vr=U[:, :svd_save])
    print("\033[1m Done.\033[0m")

    print("\033[1m Projecting data...\033[0m")
    Xhat = Q_train.T @ U
    np.save(Xhat_file, Xhat)
    print("\033[1m Done.\033[0m")
