# %%
from pathlib import Path

import latexplotlib as lpl
import matplotlib as mpl
import numpy as np
import xarray as xr

from opinf_for_hw import config as cfg
from opinf_for_hw.frequency_analysis import (
    power_spectrum,
    wasserstein_fourier_metric,
)

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

lpl.size.set(246, 569)

# %%
DATA_DIR = Path("../../data")
RANKS = [60, 138]
R1 = RANKS[0]
R2 = RANKS[1]
FIGURE_PATH = Path("figures")

# %% load data
fh = xr.open_dataset(DATA_DIR / "reference/invariants.h5")
Gamma_n_ref = fh[r"$\Gamma_n$"].data[20000:40001]
Gamma_c_ref = fh[r"$\Gamma_c$"].data[20000:40001]

Gamma_n_OpInf_r1 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_n_pred_quad_training_end{cfg.TRAINING_END}_r{R1}.npy"
)
Gamma_n_OpInf_r2 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_n_pred_quad_training_end{cfg.TRAINING_END}_r{R2}.npy"
)

Gamma_n_train = {
    R1: Gamma_n_OpInf_r1[: cfg.TRAINING_END],
    R2: Gamma_n_OpInf_r2[: cfg.TRAINING_END],
}
Gamma_n_pred = {
    R1: Gamma_n_OpInf_r1[cfg.TRAINING_END :],
    R2: Gamma_n_OpInf_r2[cfg.TRAINING_END :],
}


Gamma_c_OpInf_r1 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_c_pred_quad_training_end{cfg.TRAINING_END}_r{R1}.npy"
)
Gamma_c_OpInf_r2 = np.load(
    DATA_DIR
    / cfg.DATA_SUBDIR
    / f"Gamma_c_pred_quad_training_end{cfg.TRAINING_END}_r{R2}.npy"
)

Gamma_c_train = {
    R1: Gamma_c_OpInf_r1[: cfg.TRAINING_END],
    R2: Gamma_c_OpInf_r2[: cfg.TRAINING_END],
}
Gamma_c_pred = {
    R1: Gamma_c_OpInf_r1[cfg.TRAINING_END :],
    R2: Gamma_c_OpInf_r2[cfg.TRAINING_END :],
}


# %% helper function
def compute_and_plot(ds, *, ax, marker, color):
    data = power_spectrum(ds, frequency=1 / cfg.DT)
    return data.plot.scatter(
        ax=ax, x="freq_time", marker=marker, lw=1, s=9, color=color
    )


# %% errors
d1 = wasserstein_fourier_metric(Gamma_n_ref[: cfg.TRAINING_END], Gamma_n_train[R1])
d2 = wasserstein_fourier_metric(Gamma_n_ref[: cfg.TRAINING_END], Gamma_n_train[R2])

print(d1, d2)


d3 = wasserstein_fourier_metric(Gamma_n_ref[cfg.TRAINING_END :], Gamma_n_pred[R1])
d4 = wasserstein_fourier_metric(Gamma_n_ref[cfg.TRAINING_END :], Gamma_n_pred[R2])

print(d3, d4)

d1 = wasserstein_fourier_metric(Gamma_c_ref[: cfg.TRAINING_END], Gamma_c_train[R1])
d2 = wasserstein_fourier_metric(Gamma_c_ref[: cfg.TRAINING_END], Gamma_c_train[R2])

print(d1, d2)


d3 = wasserstein_fourier_metric(Gamma_c_ref[cfg.TRAINING_END :], Gamma_c_pred[R1])
d4 = wasserstein_fourier_metric(Gamma_c_ref[cfg.TRAINING_END :], Gamma_c_pred[R2])

print(d3, d4)

# %% figure
with lpl.size.context(510, 672):
    fig, (laxes, *axes) = lpl.subplots(
        3, 2, aspect=2.5, sharex=True, sharey=True, height_ratios=[0.05, 1.0, 1.0]
    )

colors = ["C0", "C1", "C2"]
markers = [".", "*", "2"]

marker = markers[0]
color = colors[0]

compute_and_plot(
    Gamma_n_ref[: cfg.TRAINING_END], ax=axes[0][0], marker=marker, color=color
)
compute_and_plot(
    Gamma_n_ref[cfg.TRAINING_END :], ax=axes[0][1], marker=marker, color=color
)

compute_and_plot(
    Gamma_c_ref[: cfg.TRAINING_END], ax=axes[1][0], marker=marker, color=color
)
compute_and_plot(
    Gamma_c_ref[cfg.TRAINING_END :], ax=axes[1][1], marker=marker, color=color
)

for r, color, marker in zip(RANKS, colors[1:], markers[1:]):
    compute_and_plot(Gamma_n_train[r], ax=axes[0][0], marker=marker, color=color)
    compute_and_plot(Gamma_n_pred[r], ax=axes[0][1], marker=marker, color=color)

    compute_and_plot(Gamma_c_train[r], ax=axes[1][0], marker=marker, color=color)
    compute_and_plot(Gamma_c_pred[r], ax=axes[1][1], marker=marker, color=color)

axes[0][0].set_ylabel(r"power spectrum $\Gamma_n$")
axes[1][0].set_ylabel(r"power spectrum $\Gamma_c$")

ax = axes[0][0]
ax.set_xscale("log")
ax.set_xlim(8e-3, 1.1)
ax.set_yscale("log")
ax.set_ylim(1e-8, 1e-1)

ax.set_xticks([1e-2, 1e-1, 1])
ax.set_xticklabels(["$10^{-2}$", "$10^{-1}$", "$10^0$"])

ax.set_yticks([10**n for n in range(-8, -1, 2)])
ax.set_yticklabels([f"$10^{{{n}}}$" for n in range(-8, -1, 2)])

for ax in axes[0]:
    ax.set_xlabel("")
    ax.xaxis.set_ticks_position("bottom")
for ax in axes:
    ax[1].yaxis.set_ticks_position("left")

axes[0][0].set_title("training time horizon")
axes[0][1].set_title("prediction time horizon")

axes[1][0].set_xlabel(r"normalized frequency $\bar{\omega} / \omega_{de}$")
axes[1][1].set_xlabel(r"normalized frequency $\bar{\omega} / \omega_{de}$")

for ax in laxes:
    ax.axis("off")

gs = laxes[0].get_gridspec()
lax = fig.add_subplot(gs[0, :])
lax.axis("off")
legend = lax.legend(
    [
        mpl.lines.Line2D([], [], ls="", marker=marker, color=color)
        for marker, color in zip(markers, colors)
    ],
    [
        "reference (N = 524K)",
        f"OpInf ROM (r = {R1})",
        f"OpInf ROM (r = {R2})",
    ],
    loc="center",
    ncol=3,
    borderaxespad=0,
)

for i, text in enumerate(legend.get_texts()):
    text.set_color(f"C{i}")

fig.savefig(FIGURE_PATH / ("HW_power_spectra.pdf"))
