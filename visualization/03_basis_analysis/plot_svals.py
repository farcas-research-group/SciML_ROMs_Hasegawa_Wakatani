# %%
from pathlib import Path

import latexplotlib as lpl
import numpy as np

from opinf_for_hw import config as cfg

# %%
lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

lpl.size.set(246, 569)

# %%
DATA_PATH = Path("../../data")
FIGURE_PATH = Path("figures")

# %%
pod = np.load(DATA_PATH / cfg.POD_FILE)
svals = pod["S"]

# %%
no_kept_svals_global = 9999
no_svals_global = range(1, no_kept_svals_global + 1)

# %%
retained_energy = np.cumsum(svals**2) / np.sum(svals**2)

# %%
ranks = [
    60,
    138,
    int(np.argmax(retained_energy > 0.90) + 1),
    int(np.argmax(retained_energy > 0.95) + 1),
    int(np.argmax(retained_energy > 0.99) + 1),
]

for rank in ranks:
    print(f"rank = {rank:3d}: energy = {retained_energy[rank - 1]:.4f}")

# %%
r_1 = ranks[0]
r_2 = ranks[1]
e_1 = retained_energy[r_1 - 1]
e_2 = retained_energy[r_2 - 1]

# %%
fig, (ax0, ax1) = lpl.subplots(1, 2, aspect=1.2)

ax1.plot([r_1, r_1], [0, e_1], lw=0.5, ls="--", color="C0")
ax1.plot([0, r_1], [e_1, e_1], lw=0.5, ls="--", color="C0")
ax1.plot([r_2, r_2], [0, e_2], lw=0.5, ls="--", color="C0")
ax1.plot([0, r_2], [e_2, e_2], lw=0.5, ls="--", color="C0")

ax0.semilogy(no_svals_global, svals[:no_kept_svals_global], ls="-")
ax0.set_xlabel("index")
ax0.set_ylabel("POD singular values")

no_kept_svals_global = 399
no_svals_global = range(1, no_kept_svals_global + 1)

retained_energy = np.cumsum(svals**2) / np.sum(svals**2)

ax1.plot(no_svals_global, retained_energy[:no_kept_svals_global], ls="-")
ax1.set_xlabel("reduced dimension")
ax1.set_ylabel("POD retained energy")

ax0.set_xlim((0, 10050))
ax0.set_ylim((1e-6, 1e4))

ax0.set_xticks([1, 2000, 6000, 10000])
ax0.set_xticklabels([1, 2000, 6000, 10000])

ax1.set_xlim((0, 400))
ax1.set_ylim((0, 1))

ax1.set_xticks([1, 60, 138, 250, 400])
ax1.set_yticks([0.2, 0.5, 0.75, 0.9020, 0.9694, 1])
ax1.set_yticklabels([r"$20\%$", r"$50\%$", r"$75\%$", r"$90.20\%$", r"$96.94\%$", ""])

fig.savefig(FIGURE_PATH / "HW_POD_svals_and_ret_energy.pdf")
