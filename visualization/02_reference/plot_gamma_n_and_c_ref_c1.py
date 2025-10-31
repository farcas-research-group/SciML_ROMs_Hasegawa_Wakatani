# %%
from pathlib import Path

import latexplotlib as lpl
import xarray as xr

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

# %%
C1S = ["0.10", "1.0", "5.0"]
ENGINE = "h5netcdf"
DATA_DIR = Path("../../data")
FIGURE_PATH = Path("figures")

END = {"0.10": 800, "1.0": 800, "5.0": 1750}

# %%
invariants = {}
for c1 in C1S:
    fh = xr.open_dataset(DATA_DIR / c1 / "raw.h5", engine="h5netcdf").loc[
        {"time": slice(0, END[c1])}
    ]

    invariants[c1] = fh

# %%
with lpl.size.context(246, 672):
    fig, (lax, *axes) = lpl.subplots(
        4, 1, aspect=3.5, height_ratios=(0.05, 1.0, 1.0, 1.0)
    )

name = ["gamma_n", "gamma_c"]
labels = [r"$\Gamma_n$", r"$\Gamma_c$"]
colors = ["C0", "C1"]


for n, c1 in enumerate(C1S):
    time = invariants[c1]["time"]
    gamma_n = invariants[c1]["gamma_n"]
    gamma_c = invariants[c1]["gamma_c"]
    gamma_n.name = ""
    gamma_c.name = ""

    line_n = axes[n].plot(time, gamma_n, lw=0.5, color=colors[0])
    line_c = axes[n].plot(time, gamma_c, lw=0.5, color=colors[1])


axes[0].set_title("")
# axes[0].set_xlabel("")
axes[0].set_xlim([-4, 804])
axes[0].set_ylim([-0.105, 3.105])
axes[0].set_yticks([0, 1, 2, 3])

axes[1].set_xlim([-4, 804])
axes[1].set_ylim([-0.028, 0.828])
axes[1].set_yticks([0, 0.4, 0.8])

axes[2].set_xlabel(r"normalized time $\bar{{t}} \omega_{{de}}$")
axes[2].set_xlim([-8.75, 1758.75])
axes[2].set_xticks(list(range(0, 1751, 250)))
axes[2].set_ylim([-0.007, 0.207])
axes[2].set_yticks([0, 0.1, 0.2])


for n, c1 in enumerate(C1S):
    axes[n].text(
        -0.12,
        0.5,
        f"$c_1 = {c1 if c1 != '0.10' else '0.1'}$",
        ha="center",
        va="center",
        rotation=90,
        transform=axes[n].transAxes,
    )

lax.axis("off")
legend = lax.legend(
    [line_n[0], line_c[0]],
    [r"$\Gamma_n$", r"$\Gamma_c$"],
    loc="center",
    ncol=2,
    borderaxespad=0.5,
)

for text, color in zip(legend.get_texts(), colors, strict=True):
    text.set_color(color)

fig.savefig(FIGURE_PATH / "HW_Gamma_n_and_c_ref_c1")
