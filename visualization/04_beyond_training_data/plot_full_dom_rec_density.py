# %%
from pathlib import Path

import latexplotlib as lpl
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from sklearn.metrics.pairwise import cosine_similarity

from opinf_for_hw import config as cfg

# %%
msg = """
    This script searches hyperparameters for predictions beyond the training data.
    However, 'MULTIPLE_IC = True', this probably means that you used the wrong
    parameters in the previous steps.

    You should change 'MULTIPLE_IC = False' and rerun scripts 2+.
"""
assert not cfg.MULTIPLE_IC, msg  # noqa: S101


# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

lpl.size.set(246, 569)

# %%
DATA_DIR = Path("../../data/")
CMAP = "RdBu_r"

R1 = 60
R2 = 138

# %%
# ref = xr.open_dataset(
#     DATA_DIR / cfg.REFERENCE_SUBDIR / "ref_snapshots.h5", engine=cfg.ENGINE
# )
X_flat = np.load(DATA_DIR / cfg.REFERENCE_SUBDIR / "ref_snapshots.npy")
data_array_r1 = np.load(DATA_DIR / cfg.DATA_SUBDIR / f"rec_snapshots_quad_r{R1}.npy")
data_array_r2 = np.load(DATA_DIR / cfg.DATA_SUBDIR / f"rec_snapshots_quad_r{R2}.npy")


# %%
indices = [0, 4, 7, -1]

density_ref_1 = X_flat[:, indices[0]][: 512 * 512].reshape(512, 512).T
density_opinf_r1_1 = data_array_r1[:, indices[0]][: 512 * 512].reshape(512, 512).T
density_opinf_r2_1 = data_array_r2[:, indices[0]][: 512 * 512].reshape(512, 512).T

density_ref_2 = X_flat[:, indices[1]][: 512 * 512].reshape(512, 512).T
density_opinf_r1_2 = data_array_r1[:, indices[1]][: 512 * 512].reshape(512, 512).T
density_opinf_r2_2 = data_array_r2[:, indices[1]][: 512 * 512].reshape(512, 512).T

density_ref_3 = X_flat[:, indices[2]][: 512 * 512].reshape(512, 512).T
density_opinf_r1_3 = data_array_r1[:, indices[2]][: 512 * 512].reshape(512, 512).T
density_opinf_r2_3 = data_array_r2[:, indices[2]][: 512 * 512].reshape(512, 512).T

density_ref_4 = X_flat[:, indices[3]][: 512 * 512].reshape(512, 512).T
density_opinf_r1_4 = data_array_r1[:, indices[3]][: 512 * 512].reshape(512, 512).T
density_opinf_r2_4 = data_array_r2[:, indices[3]][: 512 * 512].reshape(512, 512).T


mean_ref_1 = np.mean(density_ref_1)
mean_opinf_r1_1 = np.mean(density_opinf_r1_1)
mean_opinf_r2_1 = np.mean(density_opinf_r2_1)

print(mean_ref_1)
print(mean_opinf_r1_1)
print(mean_opinf_r2_1)

print("*****************")


mean_ref_2 = np.mean(density_ref_2)
mean_opinf_r1_2 = np.mean(density_opinf_r1_2)
mean_opinf_r2_2 = np.mean(density_opinf_r2_2)

print(mean_ref_2)
print(mean_opinf_r1_2)
print(mean_opinf_r2_2)


print("*****************")
mean_ref_3 = np.mean(density_ref_3)
mean_opinf_r1_3 = np.mean(density_opinf_r1_3)
mean_opinf_r2_3 = np.mean(density_opinf_r2_3)

print(mean_ref_3)
print(mean_opinf_r1_3)
print(mean_opinf_r2_3)


print("*****************")
mean_ref_4 = np.mean(density_ref_4)
mean_opinf_r1_4 = np.mean(density_opinf_r1_4)
mean_opinf_r2_4 = np.mean(density_opinf_r2_4)

print(mean_ref_4)
print(mean_opinf_r1_4)
print(mean_opinf_r2_4)

print("*****************")
cov_1 = cosine_similarity(density_ref_1, density_opinf_r1_1)
cov_2 = cosine_similarity(density_ref_1, density_opinf_r2_1)

print(cov_1)
print(cov_2)

print("*****************")
cov_1 = cosine_similarity(density_ref_2, density_opinf_r1_2)
cov_2 = cosine_similarity(density_ref_2, density_opinf_r2_2)

print(cov_1)
print(cov_2)

print("*****************")
cov_1 = cosine_similarity(density_ref_3, density_opinf_r1_3)
cov_2 = cosine_similarity(density_ref_3, density_opinf_r2_3)

print(cov_1)
print(cov_2)

print("*****************")
cov_1 = cosine_similarity(density_ref_4, density_opinf_r1_4)
cov_2 = cosine_similarity(density_ref_4, density_opinf_r2_3)

print(cov_1)
print(cov_2)

# %%
with lpl.rc_context(
    {
        "axes.spines.left": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
):
    figsize = lpl.figsize(3, 4, aspect=0.81, width_ratios=(1.0, 1.0, 1.0, 0.1))
    fig = lpl.figure(figsize=figsize)

    grid = ImageGrid(
        fig=fig,
        rect=(0.13, 0.075, 0.80, 0.88),
        nrows_ncols=(4, 3),
        share_all=True,
        axes_pad=0.12,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.06,
        cbar_size="2%",
    )

clim = 8
extent = (-cfg.L / 2, cfg.L / 2, -cfg.L / 2, cfg.L / 2)

grid[0].imshow(density_ref_1, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[1].imshow(density_opinf_r1_1, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[2].imshow(density_opinf_r2_1, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)

grid[3].imshow(density_ref_2, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[4].imshow(density_opinf_r1_2, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[5].imshow(density_opinf_r2_2, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)

grid[6].imshow(density_ref_3, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[7].imshow(density_opinf_r1_3, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[8].imshow(density_opinf_r2_3, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)

grid[9].imshow(density_ref_4, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[10].imshow(density_opinf_r1_4, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)
grid[11].imshow(density_opinf_r2_4, vmin=-clim, vmax=clim, cmap=CMAP, extent=extent)

for ax, time in zip(grid.axes_all[::3], [1000, 8000, 12000, 20000]):
    ax.yaxis.set_ticks_position("left")

    label = rf"$\bar{{t}} \omega_\text{{de}} =  {500 + int(time * cfg.DT)}$"
    ax.text(-41, 0, label, ha="center", va="center", rotation=90)
    ax.text(-33, 0, r"$\bar{{y}}/\rho_s$", ha="center", va="center", rotation=90)

for ax in grid.axes_all[9::]:
    ax.text(0, -33, r"$\bar{{x}}/\rho_s$", ha="center", va="center")


for ax in grid.axes_all[9::]:
    ax.xaxis.set_ticks_position("bottom")

grid[0].set_xticks([-cfg.L / 2, 0, cfg.L / 2])
grid[0].set_xticklabels(["-$21$", "$0$", "$21$"])

grid[0].set_yticks([-cfg.L / 2, 0, cfg.L / 2])
grid[0].set_yticklabels(["-$21$", "$0$", "$21$"])

grid[0].set_title(r"ref. (N = 524K)")
grid[1].set_title(r"ROM (r = 60)")
grid[2].set_title(r"ROM (r = 138)")

cax = grid.cbar_axes[0]
norm = mpl.colors.Normalize(-clim, clim, clip=True)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)

fig.savefig("figures/HW_density_rec_quad_OpInf.pdf")
