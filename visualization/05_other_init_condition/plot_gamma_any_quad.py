# %%
from pathlib import Path

import latexplotlib as lpl
import numpy as np
import xarray as xr

from opinf_for_hw import config as cfg

# %%
msg = """
    This script visualizes the invariants for multiple initial conditions. However,
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

R1 = 60
R2 = 138

T_REF = np.arange(20000 * cfg.DT, 30000 * cfg.DT, cfg.DT)
IDMAP = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7}

# %% load data
ref: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}
preds_r1: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}
preds_r2: dict[str, dict[int, np.ndarray]] = {"gamma_n": {}, "gamma_c": {}}

for key, id_ in IDMAP.items():
    path = Path(DATA_DIR / cfg.DATA_SUBDIR / f"hw_invariants_{id_}.h5", exists=True)
    fh = xr.open_dataset(path, engine=cfg.ENGINE)

    ref["gamma_n"][key] = fh[r"$\Gamma_n$"].data[20000:30000]

    path = Path(DIR / "Gamma_pred_all_init_cond" / f"Gamma_n_pred_{id_}_r{R1}.npy")
    preds_r1["gamma_n"][key] = np.load(path)

    path = Path(DIR / "Gamma_pred_all_init_cond" / f"Gamma_n_pred_{id_}_r{R2}.npy")
    preds_r2["gamma_n"][key] = np.load(path)

    ref["gamma_c"][key] = fh[r"$\Gamma_c$"].data[20000:30000]

    path = Path(DIR / "Gamma_pred_all_init_cond" / f"Gamma_c_pred_{id_}_r{R1}.npy")
    preds_r1["gamma_c"][key] = np.load(path)

    path = Path(DIR / "Gamma_pred_all_init_cond" / f"Gamma_c_pred_{id_}_r{R2}.npy")
    preds_r2["gamma_c"][key] = np.load(path)

# %% errors
for c_or_n in ["c", "n"]:
    print(f"errors for r = {R1}")
    for key, id_ in IDMAP.items():
        _ref = ref[f"gamma_{c_or_n}"][key]
        data = preds_r1[f"gamma_{c_or_n}"][key]
        print(
            f"\033[1m Statistics Gamma_{c_or_n} IC{id_} training: "
            f"ref mean   = {_ref.mean():.5f}, opinf mean   = {data.mean():.5f}\033[0m"
            "\n"
            f"\033[1m Statistics Gamma_{c_or_n} IC{id_} training: "
            f"ref stddev = {_ref.std():.5f}, opinf stddev = {data.std():.5f}\033[0m"
            "\n"
        )
    print("************************\n")

    print(f"errors for r = {R2}")
    for key, id_ in IDMAP.items():
        _ref = ref[f"gamma_{c_or_n}"][key]
        data = preds_r2[f"gamma_{c_or_n}"][key]
        print(
            f"\033[1m Statistics Gamma_{c_or_n} IC{id_} training: "
            f"ref mean   = {_ref.mean():.5f}, opinf mean   = {data.mean():.5f}\033[0m"
            "\n"
            f"\033[1m Statistics Gamma_{c_or_n} IC{id_} training: "
            f"ref stddev = {_ref.std():.5f}, opinf stddev = {data.std():.5f}\033[0m"
            "\n"
        )

    print("************************")

# %% figure
for c_or_n in ["c", "n"]:
    with lpl.size.context(510, 672):
        fig, (laxes, *axes) = lpl.subplots(
            4,
            2,
            aspect=2.5,
            sharex=True,
            sharey=True,
            height_ratios=[0.05, 1.0, 1.0, 1.0],
        )

    axes = np.concatenate(axes)

    for key, ax in zip(IDMAP.keys(), axes, strict=True):
        ax.set_title(rf"new IC $\# {key}$")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        l0 = ax.plot(T_REF, ref[f"gamma_{c_or_n}"][key])
        l1 = ax.plot(T_REF, preds_r1[f"gamma_{c_or_n}"][key])
        l2 = ax.plot(T_REF, preds_r2[f"gamma_{c_or_n}"][key])

    axes[4].set_xlabel(r"normalized time $\bar{t} \omega_{de}$")
    axes[5].set_xlabel(r"normalized time $\bar{t} \omega_{de}$")
    # fig.supxlabel(r"normalized time $\bar{t} \omega_{de}$")
    fig.supylabel(rf"$\Gamma_{c_or_n}$", fontsize=8)

    axes[0].set_xticks([500, 625, 750])
    axes[0].set_xlim(497, 753)

    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8])
    # axes[0].set_ylim(0.46, 0.81)
    axes[0].set_ylim(0.45, 0.77)

    for lax in laxes:
        lax.axis("off")

    gs = laxes[0].get_gridspec()
    lax = fig.add_subplot(gs[0, :])
    lax.axis("off")

    legend = lax.legend(
        (l0[0], l1[0], l2[0]),
        (
            "reference (N = 524K)",
            f"OpInf ROM (r = {R1})",
            f"OpInf ROM (r = {R2})",
        ),
        loc="center",
        ncol=3,
        borderaxespad=0.0,
    )

    colors = ["C0", "C1", "C2"]
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])

    fig.savefig(f"figures/HW_Gamma_{c_or_n}_other_init_cond.pdf")
