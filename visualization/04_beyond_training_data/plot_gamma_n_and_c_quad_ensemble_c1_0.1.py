# from config.HW import *
import latexplotlib as lpl
import numpy as np

lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")


def truncate(number: float, digits: int) -> float:
    nbDecimals = len(str(number).split(".")[1])

    if nbDecimals <= digits:
        return number

    stepper = 10.0**digits

    return np.trunc(stepper * number) / stepper  # type: ignore[no-any-return]


dt = 2.0e-2
t_traing_end = 400
t_ref = np.arange(300, 800, dt)

r = 78
training_size = 5000
total_size = 25000


ENGINE = "h5netcdf"

data = np.load("data/Gamma_ref_c1_0.10.npz")

Gamma_n_ref = data["Gamma_n"][:total_size]
Gamma_c_ref = data["Gamma_c"][:total_size]


data = np.load(
    "results/ensemble_c1_0.1/Gamma_ensemble_statistics_c1_0.1_training_end"
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
    "results/ensemble_c1_0.1/postprocessing_ensemble_c1_5.0_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz"
)

Gamma_n_best = data["Gamma_n_pred"]
Gamma_c_best = data["Gamma_c_pred"]


data = np.load(
    "results/ensemble_c1_0.1/Gamma_bad_reg_c1_0.1_training_end"
    + str(training_size)
    + "_r"
    + str(r)
    + ".npz"
)

Gamma_n_bad_reg = data["Gamma_n"]
Gamma_c_bad_reg = data["Gamma_c"]


# lpl.rcParams["figure.figsize"] = (8, 5)

with lpl.size.context(510, 672):
    fig, (lax, ax0, ax1) = lpl.subplots(
        3, 1, aspect=3.5, scale=1.0, height_ratios=[0.1, 1.0, 1.0], sharex=False
    )

ax0.axvline(t_traing_end, linestyle="--", lw=0.5, color="k")
ax1.axvline(t_traing_end, linestyle="--", lw=0.5, color="k")

ax0.set_xlim([297.5, 802.5])
ax0.set_ylim([0.9755, 2.4245])

ax1.set_xlim([297.5, 802.5])
# ax1.set_ylim([0.7825, 1.8175])
ax1.set_ylim([0.88425, 1.81575])

xticks = [300, 400, 500, 600, 700, 800]
xticklabels = ["$300$", "$400$\ntraining ends here", "$500$", "$600$", "$700$", "$800$"]
ax0.set_xticks(xticks)
ax0.set_xticklabels(xticklabels)
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)
ax1.set_xlabel("normalized time " + r"$\bar{t} \omega_{\mathrm{de}}$")

ax0.set_ylabel(r"$\Gamma_n$")
ax0.set_yticks([1.0, 1.35, 1.7, 2.05, 2.4])
ax1.set_ylabel(r"$\Gamma_c$")

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
p4 = ax0.plot(t_ref, Gamma_n_bad_reg, linestyle="-", lw=1.0, color="C3")

gamma_n_ref_mean = np.mean(Gamma_n_ref[training_size:])
gamma_n_ref_std = np.std(Gamma_n_ref[training_size:], ddof=1)
ax0.text(410, 2.35, "mean ref", color="C0")
ax0.text(450, 2.35, f"= {gamma_n_ref_mean:.2f}", color="C0")
ax0.text(410, 2.20, "std ref", color="C0")
ax0.text(450, 2.20, f"= {gamma_n_ref_std:.2f}", color="C0")

opinf_ensemble_mean = np.mean(Gamma_n_OpInf_mean[training_size:])
opinf_ensemble_std = np.std(Gamma_n_OpInf_mean[training_size:], ddof=1)
ax0.text(485, 2.35, "mean OpInf ensemble", color="C1")
ax0.text(575, 2.35, f"= {opinf_ensemble_mean:.2f}", color="C1")
ax0.text(485, 2.20, "std OpInf ensemble", color="C1")
ax0.text(575, 2.20, f" = {opinf_ensemble_std:.2f}", color="C1")

opinf_ensemble_mean_min_err = np.mean(Gamma_n_best[training_size:])
opinf_ensemble_std_min_err = np.std(Gamma_n_best[training_size:], ddof=1)
ax0.text(610, 2.35, "mean OpInf ensemble min err", color="C2")
ax0.text(735, 2.35, f"= {opinf_ensemble_mean_min_err:.2f}", color="C2")
ax0.text(610, 2.20, "std OpInf ensemble min err", color="C2")
ax0.text(735, 2.20, f" = {opinf_ensemble_std_min_err:.2f}", color="C2")


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
ax1.plot(t_ref, Gamma_c_bad_reg, linestyle="-", lw=1.0, color="C3")


gamma_c_ref_mean = np.mean(Gamma_c_ref[training_size:])
gamma_c_ref_std = np.std(Gamma_c_ref[training_size:], ddof=1)
ax1.text(410, 1.75, "mean ref", color="C0")
ax1.text(450, 1.75, f"= {gamma_c_ref_mean:.2f}", color="C0")
ax1.text(410, 1.65, "std ref", color="C0")
ax1.text(450, 1.65, f"= {gamma_c_ref_std:.2f}", color="C0")

opinf_ensemble_mean = np.mean(Gamma_c_OpInf_mean[training_size:])
opinf_ensemble_std = np.std(Gamma_c_OpInf_mean[training_size:], ddof=1)
ax1.text(485, 1.75, "mean OpInf ensemble", color="C1")
ax1.text(575, 1.75, f"= {opinf_ensemble_mean:.2f}", color="C1")
ax1.text(485, 1.65, "std OpInf ensemble", color="C1")
ax1.text(575, 1.65, f" = {opinf_ensemble_std:.2f}", color="C1")

opinf_ensemble_mean_min_err = np.mean(Gamma_c_best[training_size:])
opinf_ensemble_std_min_err = np.std(Gamma_c_best[training_size:], ddof=1)
ax1.text(610, 1.75, "mean OpInf ensemble min err", color="C2")
ax1.text(735, 1.75, f"= {opinf_ensemble_mean_min_err:.2f}", color="C2")
ax1.text(610, 1.65, "std OpInf ensemble min err", color="C2")
ax1.text(735, 1.65, f" = {opinf_ensemble_std_min_err:.2f}", color="C2")


# bb = (
#     fig.subplotpars.left + 0.03,
#     fig.subplotpars.top + 0.035,
#     fig.subplotpars.right - fig.subplotpars.left,
#     0.1,
# )

lax.axis("off")
legend = lax.legend(
    (p1[0], p2[0], p3[0], p4[0]),
    (
        r"reference ($N_\text{state} = 524K$)",
        f"OpInf ensemble ($r = {r}$)",
        f"OpInf ensemble min err ($r = {r}$)",
        f"OpInf min train err ($r = {r}$)",
    ),
    # bbox_to_anchor=bb,
    mode="expand",
    loc="center",
    ncol=2,
    borderaxespad=4,
    # bbox_transform=fig.transFigure,
)

colors = ["C0", "C1", "C2", "C3"]
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])


fig.savefig("figures/HW_Gamma_n_and_c_ensemble_train_5000_c1_0.1.pdf")
