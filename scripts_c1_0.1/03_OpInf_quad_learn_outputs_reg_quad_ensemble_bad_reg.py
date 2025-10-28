from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

from config.HW import *

import xarray as xr

r = 78


def get_rel_err(A, B, opt="max-rel-2"):
    if opt == "max-rel-2":
        error = np.max(np.sqrt(np.sum((B - A) ** 2, axis=1) / np.sum(A**2, axis=1)))
    elif opt == "er1":
        # entry-wise relative error
        error = np.linalg.sum(np.abs((B - A) / A)) / np.size(A)
    elif opt == "mean-rel-2":
        error = np.mean(np.sqrt(np.sum((B - A) ** 2, axis=1) / np.sum(A**2, axis=1)))
    return error


def solve_opinf_difference_model(s0, n_steps, f):
    s = np.zeros((np.size(s0), n_steps))
    is_nan = False

    s[:, 0] = s0
    for i in range(n_steps - 1):
        s[:, i + 1] = f(s[:, i])

        if np.any(np.isnan(s[:, i + 1])):
            print("NaN encountered at iteration " + str(i + 1))

            is_nan = True

            break

    return is_nan, s


###########################
print(
    "\033[1m Prepare the data for the least-squares learning procedure for the state \033[0m"
)
Xhatmax = np.load(Xhat_file)
Xhat = Xhatmax[:, :r]

X_state = Xhat[:-1, :]
Y_state = Xhat[1:, :]

s = int(r * (r + 1) / 2)

d_state = r + s
d_out = r + s + 1

X_state2 = get_x_sq(X_state)

D_state = np.concatenate((X_state, X_state2), axis=1)
D_state_2 = D_state.T @ D_state
print("\033[1m Done \033[0m")
###########################

print(
    "\033[1m Prepare the data for the least-squares learning procedure for the output \033[0m"
)
Xhatmax = np.load(Xhat_file)

X_out = Xhatmax[:, :r]
K = X_out.shape[0]
E = np.ones((K, 1))

mean_Xhat = np.mean(X_out, axis=0)
Xhat_out = X_out - mean_Xhat[np.newaxis, :]

local_min = np.min(X_out)
local_max = np.max(X_out)
local_scaling = np.maximum(np.abs(local_min), np.abs(local_max))

scaling_Xhat = local_scaling

Xhat_out /= scaling_Xhat

Xhat_out2 = get_x_sq(Xhat_out)


D_out = np.concatenate((Xhat_out, Xhat_out2, E), axis=1)
D_out_2 = D_out.T @ D_out

print("\033[1m Done \033[0m")

dy = 2

print("\033[1m BEGIN \033[0m")

print("\033[1m Transforming derived quantities for training  \033[0m")
ENGINE = "h5netcdf"
fh = xr.open_dataset("/storage1/HW/paper/0.10_300_training.h5", engine=ENGINE)

Gamma_n = fh["gamma_n"].data
Gamma_c = fh["gamma_c"].data

Y_Gamma = np.vstack((Gamma_n, Gamma_c))

print("\033[1m Done \033[0m")

print("\033[1m Search for the best regularization parameters \033[0m")


avg_err_best = 1e20

alpha_state_lin_best = 1e20
alpha_state_quad_best = 1e20

alpha_out_lin_best = 1e20
alpha_out_quad_best = 1e20

for alpha_state_lin in ridge_alf_lin_all:
    for alpha_state_quad in ridge_alf_quad_all:
        print("alpha_lin = %.2E" % alpha_state_lin)
        print("alpha_quad = %.2E" % alpha_state_quad)

        regg = np.zeros(d_state)
        regg[:r] = alpha_state_lin
        regg[r : r + s] = alpha_state_quad
        regularizer = np.diag(regg)
        D_state_reg = D_state_2 + regularizer
        # print('condition number of D.T D + reg = ', np.linalg.cond(D_reg))

        O = np.linalg.solve(D_state_reg, np.dot(D_state.T, Y_state)).T

        A = O[:, :r]
        F = O[:, r : r + s]
        f = lambda x: np.dot(A, x) + np.dot(F, get_x_sq(x))

        u0 = X_state[0, :]
        is_nan, Xhat_rk2 = solve_opinf_difference_model(u0, n_steps, f)

        X_OpInf_full = Xhat_rk2.T

        Xhat_OpInf_full = (X_OpInf_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
        Xhat_2_OpInf_full = get_x_sq(Xhat_OpInf_full)

        if not is_nan:
            for n, alpha_out_lin in enumerate(gamma_reg_lin):
                for m, alpha_out_quad in enumerate(gamma_reg_quad):
                    print(
                        "\033[1m Calculations for regularization parameters {}, {} \033[0m".format(
                            alpha_out_lin, alpha_out_quad
                        )
                    )

                    regg = np.zeros(d_out)
                    regg[:r] = alpha_out_lin
                    regg[r : r + s] = alpha_out_quad
                    regg[r + s :] = alpha_out_lin
                    regularizer = np.diag(regg)
                    D_out_reg = D_out_2 + regularizer

                    O = np.linalg.solve(D_out_reg, np.dot(D_out.T, Y_Gamma.T)).T

                    C = O[:, :r]
                    G = O[:, r : r + s]
                    c = O[:, r + s]

                    Y_OpInf = (
                        C @ Xhat_OpInf_full.T
                        + G @ Xhat_2_OpInf_full.T
                        + c[:, np.newaxis]
                    )

                    ts_Gamma_n = Y_OpInf[0, :]
                    ts_Gamma_c = Y_OpInf[1, :]

                    err_Gamma_n_train = np.linalg.norm(
                        ts_Gamma_n[:training_end] - Gamma_n
                    ) / np.linalg.norm(Gamma_n)
                    err_Gamma_c_train = np.linalg.norm(
                        ts_Gamma_c[:training_end] - Gamma_n
                    ) / np.linalg.norm(Gamma_c)

                    avg_err_train = 0.5 * (err_Gamma_n_train + err_Gamma_c_train)

                    if avg_err_best > avg_err_train:
                        avg_err_best = avg_err_train

                        alphas_lin_temp = [alpha_state_lin, alpha_out_lin]
                        alphas_quad_temp = [alpha_state_quad, alpha_out_quad]

                        alpha_state_lin_best = alpha_state_lin
                        alpha_state_quad_best = alpha_state_quad

                        alpha_out_lin_best = alpha_out_lin
                        alpha_out_quad_best = alpha_out_quad

                    print(
                        "\033[1m Average training error: {} \033[0m".format(
                            avg_err_train
                        )
                    )
                    print(
                        "\033[1m Best training error: {} \033[0m".format(avg_err_best)
                    )

print("\033[1m Done \033[0m")

print(alpha_state_lin_best, alpha_state_quad_best)
print(alpha_out_lin_best, alpha_out_quad_best)

np.savez(
    "results/bad_reg_params_c1_0.1_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz",
    alpha_state_lin_best=alpha_state_lin_best,
    alpha_state_quad_best=alpha_state_quad_best,
    alpha_out_lin_best=alpha_out_lin_best,
    alpha_out_quad_best=alpha_out_quad_best,
)
