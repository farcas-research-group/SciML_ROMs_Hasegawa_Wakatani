from opinf_for_hw.data_proc import *
from opinf_for_hw.postproc import *

import xarray as xr

from config.HW import *

if __name__ == "__main__":
    ranks = [1000, 2000, 3000]

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

    init_cond = Q_train[:, 0]
    X = Q_train[:, :-1]
    Y = Q_train[:, 1:]

    print("\033[1m Compute POD basis...\033[0m")
    U_all, S_all, V_all = np.linalg.svd(X, full_matrices=False)
    print("\033[1m Done.\033[0m")

    for rank in ranks:
        print("\nComputations for rank = ", rank)

        U = U_all[:, :rank]
        V = V_all.conj().T[:, :rank]
        S = S_all[:rank]

        temp = Y @ V @ np.diag(S ** (-1))
        A_hat = U.conj().T @ temp

        print(A_hat)

        eigs, _ = np.linalg.eig(A_hat)

        print(eigs)

        print("***********************")
