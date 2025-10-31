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
ENGINE = "h5netcdf"
DATA_DIR = Path("../../data/reference")
FIGURE_PATH = Path("figures")
TIME = 600

CMAP = "RdBu_r"

# %%
fh_ref = xr.open_dataset(DATA_DIR / "ref_snapshots.h5", engine=ENGINE)
snapshots_ref = fh_ref["data"].loc[{"time": TIME}]

# %%
with (
    lpl.size.context(246, 569),
    lpl.rc_context(
        {
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "ytick.left": False,
        }
    ),
):
    figsize = lpl.figsize(1, 4, aspect=0.79, width_ratios=(1.0, 1.0, 1.0, 0.04))
    fig = lpl.figure(figsize=figsize)
    grid = ImageGrid(
        fig=fig,
        rect=(0.09, 0.04, 0.84, 1),
        nrows_ncols=(1, 3),
        share_all=True,
        axes_pad=0.12,
        cbar_mode="edge",
        cbar_location="right",
        cbar_pad=0.06,
    )

clim = 8
for n, field in enumerate(["density", "potential"]):
    ds = snapshots_ref.loc[{"field": field}]
    ds.plot.imshow(ax=grid[n], add_colorbar=False, cmap=CMAP, vmin=-clim, vmax=clim)

ds = 0.6 * clim * snapshots_ref.loc[{"field": "vorticity"}] / max(-ds.min(), ds.max())
im = ds.plot.imshow(ax=grid[2], add_colorbar=False, cmap=CMAP, vmin=-clim, vmax=clim)

grid[0].set_title(r"density $\tilde{n}$")
grid[1].set_title(r"potential $\tilde{\phi}$")
grid[2].set_title(r"vorticity $\nabla^2 \tilde{\phi}$")

L = 2 * np.pi / 0.15

grid[0].set_xticks([-L / 2, 0, L / 2])
grid[0].set_xticklabels(["-$21$", "$0$", "$21$"])

grid[0].set_yticks([-L / 2, 0, L / 2])
grid[0].set_yticklabels(["-$21$", "$0$", "$21$"])

grid[0].yaxis.set_ticks_position("left")
grid[0].text(-33, 0, r"$\bar{{y}}/\rho_s$", ha="center", va="center", rotation=90)

for ax in grid:
    ax.text(0, -33, r"$\bar{{x}}/\rho_s$", ha="center", va="center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)

cax = grid.cbar_axes[0]
norm = mpl.colors.Normalize(-clim, clim, clip=True)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cbar.set_ticks([-8, -4, 0, 4, 8])

lpl.savefig(FIGURE_PATH / "HW_snapshots_ref")
