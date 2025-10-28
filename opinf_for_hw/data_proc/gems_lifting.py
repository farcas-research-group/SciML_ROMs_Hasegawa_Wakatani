import numpy as np


def parse_vars(snapshots, n_cells, rep):
    if len(np.shape(snapshots)) == 1:  # if vector pretend it's a matrix
        snapshots = np.array([snapshots]).T

    if rep == "raw":
        U = snapshots[0 * n_cells : 1 * n_cells, :]
        V = snapshots[1 * n_cells : 2 * n_cells, :]
        W = snapshots[2 * n_cells : 3 * n_cells, :]
        p = snapshots[3 * n_cells : 4 * n_cells, :]
        T = snapshots[4 * n_cells : 5 * n_cells, :]
        rho = snapshots[5 * n_cells : 6 * n_cells, :]
        O2 = snapshots[6 * n_cells : 7 * n_cells, :]
        CH4 = snapshots[7 * n_cells : 8 * n_cells, :]

        return U, V, W, p, T, rho, O2, CH4

    elif rep == "lifted":
        U = snapshots[0 * n_cells : 1 * n_cells, :]
        V = snapshots[1 * n_cells : 2 * n_cells, :]
        W = snapshots[2 * n_cells : 3 * n_cells, :]
        p = snapshots[3 * n_cells : 4 * n_cells, :]
        T = snapshots[4 * n_cells : 5 * n_cells, :]
        xi = snapshots[5 * n_cells : 6 * n_cells, :]
        O2 = snapshots[6 * n_cells : 7 * n_cells, :]
        CH4 = snapshots[7 * n_cells : 8 * n_cells, :]

        return U, V, W, p, T, xi, O2, CH4

    else:
        print("Invalid data form")


def lift_raw_data(raw_snapshots, n_cells):
    U, V, W, p, T, rho, O2, CH4 = parse_vars(raw_snapshots, n_cells, "raw")

    xi = rho ** (-1)

    lifted_snapshots = np.concatenate((U, V, W, p, T, xi, O2, CH4), axis=0)

    return lifted_snapshots
