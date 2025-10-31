# %%
from pathlib import Path

import latexplotlib as lpl
import matplotlib as mpl
import numpy as np
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
L = 2 * np.pi / 0.15
DX = L / 512

CMAP = "RdBu_r"

MODE_IDX = [0, 3, 11, 43]
MODE_IDX = [0, 9, 19, 43]

# %%
bases = {
    c1: np.load(DATA_DIR / c1 / "no_scale/POD.npz")["Vr"][:, MODE_IDX] for c1 in C1S
}

# %%
with (
    lpl.size.context(510, 672),
    lpl.rc_context(
        {
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            # "xtick.bottom": False,
            # "ytick.left": False,
        }
    ),
):
    figsize = lpl.figsize(1, 1, aspect=1.5)
    fig = lpl.figure(figsize=figsize)
    grid = ImageGrid(
        fig=fig,
        rect=(0.08, 0.0, 0.85, 0.99),
        nrows_ncols=(4, 6),
        share_all=True,
        axes_pad=0.12,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.06,
        cbar_size="3%",
        label_mode="L",
    )

clim = np.percentile(np.abs(np.concatenate(list(bases.values()))), 99)

for n_c1, basis in enumerate(bases.values()):
    for row in range(4):
        _basis = basis[:, row].reshape((2, 512, 512))
        for n in range(2):
            grid[6 * row + 2 * n_c1 + n].imshow(
                _basis[n],
                cmap=CMAP,
                vmin=-clim,
                vmax=clim,
                extent=(
                    -L / 2 - DX / 2,
                    L / 2 - DX / 2,
                    -L / 2 - DX / 2,
                    L / 2 - DX / 2,
                ),
            )

for ax in grid:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)


grid[18].set_xticks([-L / 2, 0, L / 2])
grid[18].set_xticks([-20, 0, 20])
grid[18].set_xticklabels(["-$21$", "$0$", "$21$"])
grid[18].set_xticklabels(["-$20$", "$0$", "$20$"])
grid[18].xaxis.set_ticks_position("bottom")

grid[18].set_yticks([-L / 2, 0, L / 2])
grid[18].set_yticks([-20, 0, 20])
grid[18].set_yticklabels(["-$21$", "$0$", "$21$"])
grid[18].set_yticklabels(["-$20$", "$0$", "$20$"])
grid[18].yaxis.set_ticks_position("left")

for n, c1 in enumerate(C1S):
    grid[2 * n].text(
        24, 34, f"$c_1 = {c1 if c1 != '0.10' else '0.1'}$", ha="center", va="center"
    )
    grid[2 * n].set_title(r"density $\tilde{n}$")
    grid[2 * n + 1].set_title(r"potential $\tilde{\phi}$")

for n in range(18, 24, 1):
    grid[n].text(0, -35, r"$\bar{{x}}/\rho_s$", ha="center", va="center")

for mode_id, n in zip(MODE_IDX, range(0, 24, 6), strict=True):
    grid[n].text(
        -39,
        0,
        f"{mode_id + 1}{'st' if mode_id == 0 else 'th'} POD mode"
        + "\n"
        + r"$\bar{{y}}/\rho_s$",
        ha="center",
        va="center",
        rotation=90,
    )


cax = grid.cbar_axes[0]
norm = mpl.colors.Normalize(-clim, clim, clip=True)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
# cbar.set_ticks([-8, -4, 0, 4, 8])

lpl.savefig(FIGURE_PATH / "HW_POD_modes_all_c1_single")
