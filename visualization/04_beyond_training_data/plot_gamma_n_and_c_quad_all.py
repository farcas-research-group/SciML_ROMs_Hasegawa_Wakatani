# %%
from pathlib import Path

import latexplotlib as lpl
import numpy as np
import xarray as xr

from opinf_for_hw import config as cfg

# %%
msg = """
    This script visualizes the QoIs for predictions beyond the training data, however,
    'MULTIPLE_IC = True'.

    You should change 'MULTIPLE_IC = False'.
"""
assert not cfg.MULTIPLE_IC, msg  # noqa: S101

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

lpl.size.set(246, 569)

# %%
DATA_DIR = Path("../../data")
T_REF = np.arange(500, 1000 + cfg.DT, cfg.DT)

R1 = 60
R2 = 138

# %%
fh = xr.open_dataset(
    DATA_DIR / cfg.REFERENCE_SUBDIR / "invariants.h5", engine=cfg.ENGINE
)

key = r"$\Gamma_n$"
Gamma_n_ref = fh[key].data[20000:40001]

key = r"$\Gamma_c$"
Gamma_c_ref = fh[key].data[20000:40001]

# %%
Gamma_c_OpInf_r1 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_c_pred_quad_training_end{cfg.TRAINING_SIZE}_r{R1}.npy"
)
Gamma_c_OpInf_r2 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_c_pred_quad_training_end{cfg.TRAINING_SIZE}_r{R2}.npy"
)

Gamma_n_OpInf_r1 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_n_pred_quad_training_end{cfg.TRAINING_SIZE}_r{R1}.npy"
)
Gamma_n_OpInf_r2 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_n_pred_quad_training_end{cfg.TRAINING_SIZE}_r{R2}.npy"
)

# %% print mean and std of results
print("\033[1m Results for Gamma_n \033[0m")
print("\033[1m Training \033[0m")
print(
    "\033[1m Reference results:          mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        np.mean(Gamma_n_ref[: cfg.TRAINING_SIZE]),
        np.std(Gamma_n_ref[: cfg.TRAINING_SIZE]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R1,
        np.mean(Gamma_n_OpInf_r1[: cfg.TRAINING_SIZE]),
        np.std(Gamma_n_OpInf_r1[: cfg.TRAINING_SIZE]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R2,
        np.mean(Gamma_n_OpInf_r2[: cfg.TRAINING_SIZE]),
        np.std(Gamma_n_OpInf_r2[: cfg.TRAINING_SIZE]),
    )
)


print("\033[1m Prediction \033[0m")
print(
    "\033[1m Reference results:          mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        np.mean(Gamma_n_ref[cfg.TRAINING_SIZE :]),
        np.std(Gamma_n_ref[cfg.TRAINING_SIZE :]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R1,
        np.mean(Gamma_n_OpInf_r1[cfg.TRAINING_SIZE :]),
        np.std(Gamma_n_OpInf_r1[cfg.TRAINING_SIZE :]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R2,
        np.mean(Gamma_n_OpInf_r2[cfg.TRAINING_SIZE :]),
        np.std(Gamma_n_OpInf_r2[cfg.TRAINING_SIZE :]),
    )
)
print("\033[1m Done \033[0m")


print("\033[1m Results for Gamma_c \033[0m")
print("\033[1m Training \033[0m")
print(
    "\033[1m Reference results:          mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        np.mean(Gamma_c_ref[: cfg.TRAINING_SIZE]),
        np.std(Gamma_c_ref[: cfg.TRAINING_SIZE]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R1,
        np.mean(Gamma_c_OpInf_r1[: cfg.TRAINING_SIZE]),
        np.std(Gamma_c_OpInf_r1[: cfg.TRAINING_SIZE]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R2,
        np.mean(Gamma_c_OpInf_r2[: cfg.TRAINING_SIZE]),
        np.std(Gamma_c_OpInf_r2[: cfg.TRAINING_SIZE]),
    )
)


print("\033[1m Prediction \033[0m")
print(
    "\033[1m Reference results:          mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        np.mean(Gamma_c_ref[cfg.TRAINING_SIZE :]),
        np.std(Gamma_c_ref[cfg.TRAINING_SIZE :]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R1,
        np.mean(Gamma_c_OpInf_r1[cfg.TRAINING_SIZE :]),
        np.std(Gamma_c_OpInf_r1[cfg.TRAINING_SIZE :]),
    )
)
print(
    "\033[1m OpInf results r = {:3d}:      mean = {:.5f}, std dev = {:.5f}\033[0m".format(
        R2,
        np.mean(Gamma_c_OpInf_r2[cfg.TRAINING_SIZE :]),
        np.std(Gamma_c_OpInf_r2[cfg.TRAINING_SIZE :]),
    )
)
print("\033[1m Done \033[0m")
print()

# %% figure
with lpl.size.context(510, 672):
    fig, (lax, ax0, ax1) = lpl.subplots(
        3, 1, aspect=4.5, sharex=True, sharey=True, height_ratios=[0.05, 1.0, 1.0]
    )

p1 = ax0.plot(T_REF, Gamma_n_ref)
p2 = ax0.plot(T_REF, Gamma_n_OpInf_r1)
p3 = ax0.plot(T_REF, Gamma_n_OpInf_r2)

ax1.plot(T_REF, Gamma_c_ref)
ax1.plot(T_REF, Gamma_c_OpInf_r1)
ax1.plot(T_REF, Gamma_c_OpInf_r2)

ax0.axvline(cfg.TRAINING_END_TIME, linestyle="--", lw=0.5, color="k")
ax1.axvline(cfg.TRAINING_END_TIME, linestyle="--", lw=0.5, color="k")


ax0.set_ylabel(r"$\Gamma_n$")
ax1.set_ylabel(r"$\Gamma_c$")
ax1.set_xlabel(r"normalized time $\bar{t} \omega_{de}$")


ax0.set_xticks([500, 625, 750, 875, 1000])
ax0.set_xticklabels(["$500$", "$625$", "$750$\ntraining ends here", "$825$", "$1000$"])
ax0.xaxis.set_ticks_position("bottom")

ax0.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
ax0.set_yticklabels(["$0.4$", "$0.5$", "$0.6$", "$0.7$", "$0.8$"])

ax0.set_xlim(497.5, 1002.5)
ax1.set_ylim(0.45, 0.77)

lax.axis("off")
legend = lax.legend(
    (p1[0], p2[0], p3[0]),
    (
        "reference (N = 524K)",
        f"OpInf ROM (r = {R1})",
        f"OpInf ROM (r = {R2})",
    ),
    loc="center",
    ncol=3,
    borderaxespad=0,
)

colors = ["C0", "C1", "C2"]
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])

fig.savefig("figures/HW_Gamma_n_and_c_quad_all.pdf")
