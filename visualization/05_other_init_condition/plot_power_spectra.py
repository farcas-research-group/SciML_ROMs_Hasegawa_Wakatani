# %%
from pathlib import Path

import latexplotlib as lpl
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from opinf_for_hw import config as cfg
from opinf_for_hw.frequency_analysis import power_spectrum

# %%
msg = """
    This script visualizes the power spectra for multiple initial conditions. However,
    'MULTIPLE_IC = False', this probably means that you used the wrong parameters in
    the previous steps.

    You should change 'MULTIPLE_IC = True'.
"""
assert cfg.MULTIPLE_IC, msg  # noqa: S101

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

lpl.size.set(246, 569)

# %%
DATA_DIR = Path("../../data")
DIR = DATA_DIR / cfg.DATA_SUBDIR
FIGURE_PATH = Path("figures")

RANKS = [60, 138]
R1 = RANKS[0]
R2 = RANKS[1]

IDS = [1, 2, 4, 5, 6, 7]
IDMAP = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7}
ID_LEFT = 1
ID_RIGHT = 5

# %%
assert ID_LEFT in IDS  # noqa: S101
assert ID_RIGHT in IDS  # noqa: S101

# %% load data
ref: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}
preds_r1: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}
preds_r2: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}

for key, id_ in IDMAP.items():
    path = Path(DATA_DIR / cfg.DATA_SUBDIR / f"hw_invariants_{id_}.h5", exists=True)
    fh = xr.open_dataset(path, engine=cfg.ENGINE)

    ref["gamma_n"][key] = fh[r"$\Gamma_n$"].data[20000:30000]

    path = Path(DIR / f"Gamma_pred_all_init_cond/Gamma_n_pred_{id_}_r{R1}.npy")
    preds_r1["gamma_n"][key] = np.load(path)

    path = Path(DIR / f"Gamma_pred_all_init_cond/Gamma_n_pred_{id_}_r{R2}.npy")
    preds_r2["gamma_n"][key] = np.load(path)

    ref["gamma_c"][key] = fh[r"$\Gamma_c$"].data[20000:30000]

    path = Path(DIR / f"Gamma_pred_all_init_cond/Gamma_c_pred_{id_}_r{R1}.npy")
    preds_r1["gamma_c"][key] = np.load(path)

    path = Path(DIR / f"Gamma_pred_all_init_cond/Gamma_c_pred_{id_}_r{R2}.npy")
    preds_r2["gamma_c"][key] = np.load(path)


# %% helper function
def compute_and_plot(ds, *, ax, marker, color):
    data = power_spectrum(ds, frequency=1 / cfg.DT)

    return data.plot.scatter(
        ax=ax, x="freq_time", marker=marker, lw=1, s=9, color=color
    )


# %% create figure
with lpl.size.context(510, 672):
    fig, (laxes, *axes) = lpl.subplots(
        3, 2, aspect=2.5, sharex=True, sharey=True, height_ratios=[0.05, 1.0, 1.0]
    )
axes = np.concatenate(axes)

colors = ["C0", "C1", "C2"]
markers = [".", "*", "2"]

labels = ["reference (N = 524K)", f"OpInf ROM (r = {R1})", f"OpInf ROM (r = {R2})"]

marker = markers[0]
color = colors[0]

compute_and_plot(ref["gamma_n"][ID_LEFT], ax=axes[0], marker=marker, color=color)
compute_and_plot(ref["gamma_n"][ID_RIGHT], ax=axes[1], marker=marker, color=color)

compute_and_plot(ref["gamma_c"][ID_LEFT], ax=axes[2], marker=marker, color=color)
compute_and_plot(ref["gamma_c"][ID_RIGHT], ax=axes[3], marker=marker, color=color)

for data, color, marker in zip([preds_r1, preds_r2], colors[1:], markers[1:]):
    compute_and_plot(data["gamma_n"][ID_LEFT], ax=axes[0], marker=marker, color=color)
    compute_and_plot(data["gamma_n"][ID_RIGHT], ax=axes[1], marker=marker, color=color)

    compute_and_plot(data["gamma_c"][ID_LEFT], ax=axes[2], marker=marker, color=color)
    compute_and_plot(data["gamma_c"][ID_RIGHT], ax=axes[3], marker=marker, color=color)

axes[0].set_ylabel(r"power spectrum $\Gamma_n$")
axes[2].set_ylabel(r"power spectrum $\Gamma_c$")
axes[0].set_yscale("log")

axes[0].set_ylim(1e-8, 1e-1)
axes[2].set_ylim(1e-8, 1e-1)

for ax in axes:
    ax.set_xlim(8e-3, 1)
    ax.set_xlabel("")

    ax.set_xscale("log")
    ax.set_title("")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

axes[0].set_title(rf"new IC $\# {ID_LEFT}$")
axes[1].set_title(rf"new IC $\# {ID_RIGHT}$")

axes[2].set_xlabel(r"normalized frequency $\bar{\omega} / \omega_{de}$")
axes[3].set_xlabel(r"normalized frequency $\bar{\omega} / \omega_{de}$")

axes[0].set_xticks([1e-2, 1e-1, 1])
axes[0].set_xticklabels(["$10^{-2}$", "$10^{-1}$", "$10^0$"])

axes[0].set_yticks([10**n for n in range(-8, -1, 2)])
axes[0].set_yticklabels([f"$10^{{{n}}}$" for n in range(-8, -1, 2)])

for ax in laxes:
    ax.axis("off")

gs = laxes[0].get_gridspec()
lax = fig.add_subplot(gs[0, :])
lax.axis("off")
legend = lax.legend(
    [
        Line2D([], [], ls="", marker=marker, color=color)
        for marker, color in zip(markers, colors)
    ],
    labels,
    loc="center",
    ncol=3,
    borderaxespad=0,
)

for i, text in enumerate(legend.get_texts()):
    text.set_color(f"C{i}")

fig.savefig(FIGURE_PATH / "HW_power_spectra.pdf")
