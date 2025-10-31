# %%
from pathlib import Path

import latexplotlib as lpl
import numpy as np

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

# %%
C1S = ["0.10", "1.0", "5.0"]
ENGINE = "h5netcdf"
DATA_DIR = Path("../../data/")
FIGURE_PATH = Path("figures")
TIME = 600
L = 2 * np.pi / 0.15
DX = L / 512

CMAP = "RdBu_r"

# %%
svals = {c1: np.load(DATA_DIR / c1 / "no_scale/POD.npz")["S"] for c1 in C1S}

colors = ["C0", "C1", "C2"]
x_index = np.arange(1, 5001)


def retained_energy(svals):
    return np.cumsum(svals**2) / np.sum(svals**2)


p95 = {key: np.argmax(retained_energy(val) > 0.95) + 1 for key, val in svals.items()}

# %%
with lpl.size.context(246, 672):
    # fig, (laxes, axes) = lpl.subplots(2, 2, height_ratios=(0.1, 1.0))
    fig, (lax, *axes) = lpl.subplots(3, 1, height_ratios=(0.1, 1.0, 1.0))

axes[1].axhline(0.95, color="grey", lw=0.5, ls="--")

lines = []
for color, c1 in zip(colors, C1S, strict=True):
    lines.append(axes[0].plot(x_index, svals[c1], color=color, lw=0.5))

    axes[1].vlines(p95[c1], ymin=0, ymax=0.95, color=color, lw=0.5, ls="--")
    axes[1].plot(x_index, retained_energy(svals[c1]), color=color, lw=0.5)


axes[0].set_xlabel("index")
axes[0].set_xlim([0, 5000])
axes[0].set_xticks([1, 2500, 5000])
axes[0].set_xticks([1, 1000, 2000, 3000, 4000, 5000])
axes[0].set_ylabel("POD singular values")
axes[0].set_yscale("log")

axes[1].set_xlabel("reduced dimension")
axes[1].set_xlim([1, 250])
# axes[1].set_xticks([*p95.values(), 75, 150])
axes[1].set_xticks([*p95.values(), 100, 175])
axes[1].set_ylabel("POD retained energy")
axes[1].set_ylim(0.2, 1.0)
axes[1].set_yticks([0.2, 0.5, 0.75, 0.95])
axes[1].set_yticklabels([r"$20 \%$", r"$50 \%$", r"$75 \%$", r"$95 \%$"])

for color, tick in zip(colors, axes[1].xaxis.get_ticklabels(), strict=False):
    tick.set_color(color)

# for ax in laxes:
#     ax.axis("off")

# gs = laxes[0].get_gridspec()
# lax = fig.add_subplot(gs[0, :])
lax.axis("off")
legend = lax.legend(
    [line[0] for line in lines],
    [f"$c_1 = {c1 if c1 != '0.10' else '0.1'}$" for c1 in C1S],
    loc="center",
    ncol=3,
    borderaxespad=0,
)

for color, text in zip(colors, legend.get_texts(), strict=True):
    text.set_color(color)

lpl.savefig(FIGURE_PATH / "HW_POD_svals_and_ret_energy_all_stack")
