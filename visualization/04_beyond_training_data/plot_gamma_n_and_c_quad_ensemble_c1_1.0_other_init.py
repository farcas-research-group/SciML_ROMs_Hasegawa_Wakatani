# from config.HW import *
import latexplotlib as lpl
import numpy as np
import xarray as xr

lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")


dt = 0.025
t_traing_end = 600
t_ref = np.arange(500, 1000.01, dt)

r = 44
training_size = 4000
total_size = 20001


ENGINE = "h5netcdf"

fh = xr.open_dataset("pred_4000/data/hw_QoI_ref_pred_over_time.h5", engine=ENGINE)
key = r"$\Gamma_n$"
Gamma_n_ref = fh[key].data[20000:40001]

key = r"$\Gamma_c$"
Gamma_c_ref = fh[key].data[20000:40001]


data = np.load(
    "pred_4000/results/ensemble_c1_1.0/Gamma_ensemble_statistics_c1_1.0_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz"
)

Gamma_n_OpInf_mean = data["Gamma_n_mean"]
Gamma_c_OpInf_mean = data["Gamma_c_mean"]

Gamma_n_OpInf_std = data["Gamma_n_std"]
Gamma_c_OpInf_std = data["Gamma_c_std"]


data = np.load(
    "pred_4000/results/ensemble_c1_1.0/postprocessing_ensemble_c1_1.0_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz"
)

Gamma_n_best = data["Gamma_n_pred"]
Gamma_c_best = data["Gamma_c_pred"]


# lpl.rcParams["figure.figsize"] = (8, 5)

with lpl.size.context(510, 672):
    fig, (lax, ax0, ax1) = lpl.subplots(
        3, 1, aspect=3.5, scale=1.0, height_ratios=[0.1, 1.0, 1.0], sharex=False
    )

ax0.axvline(t_traing_end, linestyle="--", lw=0.5, color="k")
ax1.axvline(t_traing_end, linestyle="--", lw=0.5, color="k")

ax0.set_xlim([497.5, 1002.5])
# ax0.set_ylim([0.442125, 0.907875])
ax0.set_ylim([0.442125, 0.907875])
ax0.set_ylim([0.493, 0.907])

ax1.set_xlim([497.5, 1002.5])
ax1.set_ylim([0.443875, 0.806125])
ax1.set_ylim([0.49475, 0.80525])
# ax1.set_ylim([0.88425, 1.81575])

xticks = [500, 600, 700, 800, 900, 1000]
xticklabels = [
    "$500$",
    "$600$\ntraining ends here",
    "$700$",
    "$800$",
    "$900$",
    "$1000$",
]
ax0.set_xticks(xticks)
ax0.set_xticklabels(xticklabels)
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)
ax1.set_xlabel("normalized time " + r"$\bar{t} \omega_{\mathrm{de}}$")

ax0.set_ylabel(r"$\Gamma_n$")
# ax0.set_yticks([1.0, 1.35, 1.7, 2.05, 2.4])
ax1.set_ylabel(r"$\Gamma_c$")

columns = (
    "reference",
    "OpInf ensemble",
    "OpInf ensemble min err",
)
rows = ("mean", "std")

p1 = ax0.plot(t_ref, Gamma_n_ref, linestyle="-", lw=1.0, color="C0")
p2 = ax0.plot(t_ref, Gamma_n_OpInf_mean, linestyle="-", lw=1.0, color="C1")
ax0.fill_between(
    t_ref,
    Gamma_n_OpInf_mean - Gamma_n_OpInf_std,
    Gamma_n_OpInf_mean + Gamma_n_OpInf_std,
    facecolor="C1",
    alpha=0.5,
)
p3 = ax0.plot(t_ref, Gamma_n_best, linestyle="-", lw=1.0, color="C2")

datasets = [
    Gamma_n_ref[training_size:],
    Gamma_n_OpInf_mean[training_size:],
    Gamma_n_best[training_size:],
]
data = [
    [f"{np.mean(x):.2f}" for x in datasets],
    [f"{np.std(x, ddof=1):.2f}" for x in datasets],
]

table0 = ax0.table(
    cellText=data,
    cellLoc="center",
    rowLabels=rows,
    rowLoc="left",
    colLabels=columns,
    colLoc="center",
    colWidths=(0.2, 0.3, 0.3, 0.35),
    loc="best",
    bbox=[0.3, 0.8, 0.45, 0.2],
    edges="",
)

ax1.plot(t_ref, Gamma_c_ref, linestyle="-", lw=1.0, color="C0")
ax1.plot(t_ref, Gamma_c_OpInf_mean, linestyle="-", lw=1.0, color="C1")
ax1.fill_between(
    t_ref,
    Gamma_c_OpInf_mean - Gamma_c_OpInf_std,
    Gamma_c_OpInf_mean + Gamma_c_OpInf_std,
    facecolor="C1",
    alpha=0.5,
)
ax1.plot(t_ref, Gamma_c_best, linestyle="-", lw=1.0, color="C2")


datasets = [
    Gamma_c_ref[training_size:],
    Gamma_c_OpInf_mean[training_size:],
    Gamma_c_best[training_size:],
]
data = [
    [f"{np.mean(x):.2f}" for x in datasets],
    [f"{np.std(x, ddof=1):.2f}" for x in datasets],
]

table1 = ax1.table(
    cellText=data,
    cellLoc="center",
    rowLabels=rows,
    rowLoc="left",
    colLabels=columns,
    colLoc="center",
    colWidths=(0.2, 0.3, 0.3, 0.35),
    loc="best",
    bbox=[0.3, 0.8, 0.45, 0.2],
    edges="",
)

table0.auto_set_font_size(False)  # noqa: FBT003
table0.set_fontsize(lpl.rcParams["font.size"])
table1.auto_set_font_size(False)  # noqa: FBT003
table1.set_fontsize(lpl.rcParams["font.size"])

for idx, col in enumerate(["C0", "C1", "C2"]):
    table0[(0, idx)]._text.set_color(col)  # noqa: SLF001
    table0[(1, idx)]._text.set_color(col)  # noqa: SLF001
    table0[(2, idx)]._text.set_color(col)  # noqa: SLF001

    table1[(0, idx)]._text.set_color(col)  # noqa: SLF001
    table1[(1, idx)]._text.set_color(col)  # noqa: SLF001
    table1[(2, idx)]._text.set_color(col)  # noqa: SLF001

lax.axis("off")
legend = lax.legend(
    (p1[0], p2[0], p3[0]),
    (
        r"reference ($N_\text{state} = 524K$)",
        f"OpInf ensemble ($r = {r}$)",
        f"OpInf ensemble min err ($r = {r}$)",
    ),
    # bbox_to_anchor=bb,
    mode="expand",
    loc="center",
    ncol=2,
    borderaxespad=4,
    # bbox_transform=fig.transFigure,
)

colors = ["C0", "C1", "C2"]
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])


fig.savefig("figures/HW_Gamma_n_and_c_ensemble_train_4000_c1_1.0.pdf")
