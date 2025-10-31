# %%
from pathlib import Path

import latexplotlib as lpl
import xarray as xr

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

# %%
ENGINE = "h5netcdf"
DATA_DIR = Path("../../data/reference")
FIGURE_PATH = Path("figures")

# %%
fh = xr.open_dataset(DATA_DIR / "invariants.h5", engine=ENGINE)

# %%
fig, ax = lpl.subplots(1, 1, aspect=2.5)

labels = [r"$\Gamma_n$", r"$\Gamma_c$"]
colors = ["C0", "C2"]
lines = []

for label, color in zip(labels, colors):
    data = fh[label]
    lines.append(data.plot(ax=ax, lw=0.5, color=color))

ax.set_xlim(-5, 1005)
ax.set_ylim(-0.012, 0.812)

ax.set_title("")
ax.set_xlabel(r"normalized time $\bar{t} \omega_{de}$")
ax.set_ylabel("QoI")
ax.set_xticks([0, 250, 500, 750, 1000])

labels = [r"$\Gamma_n$", r"$\Gamma_c$"]
legend = ax.legend(
    [line[0] for line in lines], labels, loc="best", ncol=2, borderaxespad=0.5
)

for text, color in zip(legend.get_texts(), colors):
    text.set_color(color)

fig.savefig(FIGURE_PATH / "HW_Gamma_n_and_c_ref")
