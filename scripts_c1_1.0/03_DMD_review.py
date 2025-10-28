from op_inf_lib.data_proc import *
from op_inf_lib.postproc import *

from config.HW import *

import xarray as xr


def solve_DMD_model(s0, n_steps, f):
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


r = 200

gamma_reg_lin = np.linspace(1e-10, 1e2, 10)
gamma_reg_quad = np.linspace(1e-4, 1e4, 10)


###########################
print(
    "\033[1m Prepare the data for the least-squares learning procedure for the state \033[0m"
)
Xhatmax = np.load(Xhat_file)
Xhat = Xhatmax[:, :r]

X_state = Xhat[:-1, :].T
Y_state = Xhat[1:, :].T

d_state = r

s = int(r * (r + 1) / 2)
d_out = r + s + 1


print("\033[1m Done \033[0m")
###########################

print(
    "\033[1m Prepare the data for the least-squares learning procedure for the output \033[0m"
)
Xhatmax = np.load(Xhat_file)

Xhat_out = Xhatmax[:, :r]
K = Xhat_out.shape[0]
E = np.ones((K, 1))


mean_Xhat = np.mean(Xhat_out, axis=0)
Xhat_out = Xhat_out - mean_Xhat[np.newaxis, :]

local_min = np.min(Xhat_out)
local_max = np.max(Xhat_out)
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
fh = xr.open_dataset("/storage1/HW/paper/1.0_300_training.h5", engine=ENGINE)

Gamma_n = fh["gamma_n"].data
Gamma_c = fh["gamma_c"].data

mean_Gamma_n_ref = np.mean(Gamma_n)
std_Gamma_n_ref = np.std(Gamma_n, ddof=1)

mean_Gamma_c_ref = np.mean(Gamma_c)
std_Gamma_c_ref = np.std(Gamma_c, ddof=1)

Y_Gamma = np.vstack((Gamma_n, Gamma_c))

print("\033[1m DMD for the state \033[0m")
A_hat_state = np.load("results/A_hat_DMD_r_" + str(r) + ".npy")
print("\033[1m Done \033[0m")

print("\033[1m make predictions \033[0m")

f = lambda x: A_hat_state @ x

u0 = X_state[:, 0]
is_nan, Xhat_DMD = solve_DMD_model(u0, n_steps, f)


X_DMD_full = Xhat_DMD.T
Xhat_DMD_full = (X_DMD_full - mean_Xhat[np.newaxis, :]) / scaling_Xhat
Xhat_2_DMD_full = get_x_sq(Xhat_DMD_full)

prec_mean = 0.20
prec_std = 5.00

Gamma_n_ensemble = []
Gamma_c_ensemble = []

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

        Y_OpInf = C @ Xhat_DMD_full.T + G @ Xhat_2_DMD_full.T + c[:, np.newaxis]

        ts_Gamma_n = Y_OpInf[0, :]
        ts_Gamma_c = Y_OpInf[1, :]

        mean_Gamma_n_OpInf_train = np.mean(ts_Gamma_n[:training_end])
        std_Gamma_n_OpInf_train = np.std(ts_Gamma_n[:training_end], ddof=1)

        mean_Gamma_c_OpInf_train = np.mean(ts_Gamma_c[:training_end])
        std_Gamma_c_OpInf_train = np.std(ts_Gamma_c[:training_end], ddof=1)

        mean_Gamma_n_OpInf_pred = np.mean(ts_Gamma_n[training_end:])
        std_Gamma_n_OpInf_pred = np.std(ts_Gamma_n[training_end:], ddof=1)

        mean_Gamma_c_OpInf_pred = np.mean(ts_Gamma_c[training_end:])
        std_Gamma_c_OpInf_pred = np.std(ts_Gamma_c[training_end:], ddof=1)

        print(
            "\033[1m means Gamma n: ref {:.4} and OpInf {:.4} \033[0m".format(
                mean_Gamma_n_ref, mean_Gamma_n_OpInf_train
            )
        )
        print(
            "\033[1m stds Gamma n: ref {:.4} and OpInf {:.4} \033[0m".format(
                std_Gamma_n_ref, std_Gamma_n_OpInf_train
            )
        )

        print("*******************")

        print(
            "\033[1m means Gamma c: ref {:.4} and OpInf {:.4} \033[0m".format(
                mean_Gamma_c_ref, mean_Gamma_c_OpInf_train
            )
        )
        print(
            "\033[1m stds Gamma c: ref {:.4} and OpInf {:.4} \033[0m".format(
                std_Gamma_c_ref, std_Gamma_c_OpInf_train
            )
        )
        print("*******************")
        print("*******************")

        mean_err_Gamma_n_train = (
            np.abs(mean_Gamma_n_ref - mean_Gamma_n_OpInf_train) / mean_Gamma_n_ref
        )
        std_err_Gamma_n_train = (
            np.abs(std_Gamma_n_ref - std_Gamma_n_OpInf_train) / std_Gamma_n_ref
        )

        mean_err_Gamma_c_train = (
            np.abs(mean_Gamma_c_ref - mean_Gamma_c_OpInf_train) / mean_Gamma_c_ref
        )
        std_err_Gamma_c_train = (
            np.abs(std_Gamma_c_ref - std_Gamma_c_OpInf_train) / std_Gamma_c_ref
        )

        mean_err_Gamma_n_pred = (
            np.abs(mean_Gamma_n_ref - mean_Gamma_n_OpInf_pred) / mean_Gamma_n_ref
        )
        std_err_Gamma_n_pred = (
            np.abs(std_Gamma_n_ref - std_Gamma_n_OpInf_pred) / std_Gamma_n_ref
        )

        mean_err_Gamma_c_pred = (
            np.abs(mean_Gamma_c_ref - mean_Gamma_c_OpInf_pred) / mean_Gamma_c_ref
        )
        std_err_Gamma_c_pred = (
            np.abs(std_Gamma_c_ref - std_Gamma_c_OpInf_pred) / std_Gamma_c_ref
        )

        if (
            mean_err_Gamma_n_train < prec_mean
            and std_err_Gamma_n_train < prec_std
            and mean_err_Gamma_c_train < prec_mean
            and std_err_Gamma_c_train < prec_std
            and mean_err_Gamma_n_pred < prec_mean
            and std_err_Gamma_n_pred < prec_std
            and mean_err_Gamma_c_pred < prec_mean
            and std_err_Gamma_c_pred < prec_std
        ):
            Gamma_n_ensemble.append(ts_Gamma_n)
            Gamma_c_ensemble.append(ts_Gamma_c)

        print(
            "\033[1m Training errors for Gamma n: {}, {} \033[0m".format(
                mean_err_Gamma_n_train, std_err_Gamma_n_train
            )
        )
        print(
            "\033[1m Training errors for Gamma c: {}, {} \033[0m".format(
                mean_err_Gamma_c_train, std_err_Gamma_c_train
            )
        )

        print(
            "\033[1m Prediction errors for Gamma n: {}, {} \033[0m".format(
                mean_err_Gamma_n_pred, std_err_Gamma_n_pred
            )
        )
        print(
            "\033[1m Prediction errors for Gamma c: {}, {} \033[0m".format(
                mean_err_Gamma_c_pred, std_err_Gamma_c_pred
            )
        )
        # print('\033[1m Std devs {}, {}. {} \033[0m'.format(np.std(Gamma_c), np.std(Gamma_c_pred[:training_size]), np.std(Gamma_c_pred[training_size:])))

print("\033[1m Done \033[0m")

print("\033[1m Done \033[0m")

Gamma_n_ensemble = np.asarray(Gamma_n_ensemble)
Gamma_c_ensemble = np.asarray(Gamma_c_ensemble)


Gamma_n_mean = np.mean(Gamma_n_ensemble, axis=0)
Gamma_n_std = np.std(Gamma_n_ensemble, ddof=1, axis=0)

Gamma_c_mean = np.mean(Gamma_c_ensemble, axis=0)
Gamma_c_std = np.std(Gamma_c_ensemble, ddof=1, axis=0)

print(Gamma_n_ensemble.shape)
print(Gamma_c_ensemble.shape)


# np.savez('results/Gamma_ensemble_statistics_c1_1.0_training_end' + str(training_size) + '_r'+str(r)+'.npz', \
#         Gamma_n_mean=Gamma_n_mean, Gamma_n_std=Gamma_n_std, Gamma_c_mean=Gamma_c_mean, Gamma_c_std=Gamma_c_std)

# np.savez('results/Gamma_ensemble_statistics_c1_0.1_training_end' + str(training_size) + '_r'+str(r)+'.npz', \
#         Gamma_n_mean=Gamma_n_mean, Gamma_n_std=Gamma_n_std, Gamma_c_mean=Gamma_c_mean, Gamma_c_std=Gamma_c_std)

np.savez(
    "results/Gamma_DMD_c1_1.0_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz",
    Gamma_n_mean=Gamma_n_mean,
    Gamma_n_std=Gamma_n_std,
    Gamma_c_mean=Gamma_c_mean,
    Gamma_c_std=Gamma_c_std,
)
