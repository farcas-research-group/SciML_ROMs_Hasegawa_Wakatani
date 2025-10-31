# %%
from pathlib import Path

import latexplotlib as lpl
import matplotlib as mpl
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

# %%
C1S = ["0.10", "1.0", "5.0"]
ENGINE = "h5netcdf"
DATA_DIR = Path("../../data/")
FIGURE_PATH = Path("figures")
TIME = 600

CMAP = "RdBu_r"

# %%
snapshots = {
    c1: xr.open_dataset(DATA_DIR / c1 / f"{c1}_snapshots.h5", engine="h5netcdf")[
        {"time": 0}
    ]
    for c1 in C1S
}

# %%
with (
    lpl.size.context(246, 672),
    lpl.rc_context(
        {
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "xtick.bottom": True,
            "ytick.left": True,
        }
    ),
):
    figsize = lpl.figsize(3, 4, aspect=1.04, width_ratios=(1.0, 1.0, 1.0, 0.04))
    fig = lpl.figure(figsize=figsize)
    grid = ImageGrid(
        fig=fig,
        rect=(0.14, 0.01, 0.77, 1),
        nrows_ncols=(3, 3),
        share_all=True,
        axes_pad=0.12,
        cbar_mode="edge",
        cbar_location="right",
        cbar_pad=0.06,
        label_mode="L",
    )

clims = {"0.10": 0, "1.0": 0, "5.0": 0}
for n_c1, (c1, snapshot) in enumerate(snapshots.items()):
    clims[c1] = 1.2 * np.percentile(np.abs(snapshot["density"]), 99)

    for n, field in enumerate(["density", "potential"]):
        ds = snapshot[field]
        ds.plot.imshow(
            ax=grid[3 * n_c1 + n],
            add_colorbar=False,
            cmap=CMAP,
            vmin=-clims[c1],
            vmax=clims[c1],
        )

    scaling = np.percentile(np.abs(snapshot["vorticity"]), 99)
    ds = clims[c1] * snapshot["vorticity"] / scaling
    im = ds.plot.imshow(
        ax=grid[3 * n_c1 + 2],
        add_colorbar=False,
        cmap=CMAP,
        vmin=-clims[c1],
        vmax=clims[c1],
    )

for ax in grid:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)

    # grid[3 * n_c1].set_title("")
    # grid[3 * n_c1 + 1].set_title("")
    # grid[3 * n_c1 + 2].set_title("")

grid[0].set_title(r"density $\tilde{n}$")
grid[1].set_title(r"potential $\tilde{\phi}$")
grid[2].set_title(r"vorticity $\nabla^2 \tilde{\phi}$")

L = 2 * np.pi / 0.15

grid[6].set_xticks([-L / 2, 0, L / 2])
grid[6].set_xticks([-20, 0, 20])
grid[6].set_xticklabels(["-$21$", "$0$", "$21$"])
grid[6].set_xticklabels(["-$20$", "$0$", "$20$"])
grid[6].xaxis.set_ticks_position("bottom")

grid[6].set_yticks([-L / 2, 0, L / 2])
grid[6].set_yticks([-20, 0, 20])
grid[6].set_yticklabels(["-$21$", "$0$", "$21$"])
grid[6].set_yticklabels(["-$20$", "$0$", "$20$"])
grid[6].yaxis.set_ticks_position("left")

for c1, n in zip(C1S, range(0, 9, 3), strict=True):
    c1_str = f"$c_1 = {c1 if c1 != '0.10' else '0.1'}$"
    grid[n].text(
        -39,
        0,
        c1_str + "\n" + r"$\bar{{y}}/\rho_s$",
        ha="center",
        va="center",
        rotation=90,
    )

for ax in grid:
    ax.text(0, -35, r"$\bar{{x}}/\rho_s$", ha="center", va="center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)

for n in range(3):
    cax = grid.cbar_axes[n]
    norm = mpl.colors.Normalize(-clims[C1S[n]], clims[C1S[n]], clip=True)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    # cbar.set_ticks([-8, -4, 0, 4, 8])

lpl.savefig(FIGURE_PATH / "HW_snapshots_ref_c1")
