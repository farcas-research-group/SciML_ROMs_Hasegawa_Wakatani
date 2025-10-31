import latexplotlib as lpl
import numpy as np

lpl.style.use("latex10pt")
lpl.style.use("../paper.mplstyle")

CMAP = "RdBu_r"
C1S = [0.1, 1.0, 5.0]

r1 = 78
r2 = 44
training_size = 5000
total_size = 25000


data = np.load(
    "results/ensemble_c1_0.1/postprocessing_ensemble_c1_5.0_training_end"
    + str(training_size)
    + "_r"
    + str(r1)
    + ".npz"
)
A_red_state_lin_c1_0_1 = data["red_op_lin_state"]
eigv_c1_0_1, _ = np.linalg.eig(A_red_state_lin_c1_0_1)

data = np.load(
    "results/ensemble_c1_1.0/postprocessing_ensemble_c1_1.0_training_end"
    + str(training_size)
    + "_r"
    + str(r1)
    + ".npz"
)
A_red_state_lin_c1_1_0 = data["red_op_lin_state"]
eigv_c1_1_0, _ = np.linalg.eig(A_red_state_lin_c1_1_0)


data = np.load(
    "results/ensemble_c1_5.0/postprocessing_ensemble_c1_5.0_training_end"
    + str(training_size)
    + "_r"
    + str(r2)
    + ".npz"
)
A_red_state_lin_c1_5_0 = data["red_op_lin_state"]
eigv_c1_5_0, _ = np.linalg.eig(A_red_state_lin_c1_5_0)


A_red_state_lin_c1_0_1_nondiag = A_red_state_lin_c1_0_1 - np.eye(r1)
A_red_state_lin_c1_1_0_nondiag = A_red_state_lin_c1_1_0 - np.eye(r1)
A_red_state_lin_c1_5_0_nondiag = A_red_state_lin_c1_5_0 - np.eye(r2)

# print(np.min(A_red_state_lin_c1_0_1_nondiag), np.max(A_red_state_lin_c1_0_1_nondiag))
# print(np.min(A_red_state_lin_c1_1_0_nondiag), np.max(A_red_state_lin_c1_1_0_nondiag))
# print(np.min(A_red_state_lin_c1_5_0_nondiag), np.max(A_red_state_lin_c1_5_0_nondiag))

theta = np.linspace(0, 2 * np.pi, 200)
unit_circle = np.exp(1j * theta)

with lpl.size.context(246, 672):
    fig, axes = lpl.subplots(3, 3, aspect=1.14, width_ratios=[1.0, 0.1, 1.0])

cticks = [-0.025, -0.0125, 0, 0.0125, 0.025]
ticks = [-1.0, 0.0, 1.0]

for row, axs in enumerate(axes):
    rank = r1 if row != 2 else r2

    axs[0].set_aspect("equal")
    axs[0].set_xlabel(f"$r = {rank}$")
    axs[0].set_xticks([])
    axs[0].set_ylabel(f"$c_1 = {C1S[row]}$\n$r = {rank}$")
    axs[0].set_yticks([])

    axs[2].grid(True, ls="--", lw=0.25)  # noqa: FBT003
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("real part")
    axs[2].set_ylabel("imaginary part")

    axs[2].set_xticks(ticks)
    axs[2].set_yticks(ticks)

clim = 0.025
p1 = axes[0][0].imshow(
    A_red_state_lin_c1_0_1_nondiag, cmap=CMAP, origin="lower", vmin=-clim, vmax=clim
)
fig.colorbar(p1, cax=axes[0][1], ticks=[-0.025, -0.0125, 0, 0.0125, 0.025])


p2 = axes[1][0].imshow(
    A_red_state_lin_c1_1_0_nondiag, cmap=CMAP, origin="lower", vmin=-clim, vmax=clim
)
fig.colorbar(p2, cax=axes[1][1], ticks=[-0.025, -0.0125, 0, 0.0125, 0.025])


p3 = axes[2][0].imshow(
    A_red_state_lin_c1_5_0_nondiag, cmap=CMAP, origin="lower", vmin=-clim, vmax=clim
)
fig.colorbar(p3, cax=axes[2][1], ticks=[-0.025, -0.0125, 0, 0.0125, 0.025])


axes[0][2].plot(unit_circle.real, unit_circle.imag, color="gray", linewidth=0.5)
axes[0][2].scatter(
    eigv_c1_0_1.real, eigv_c1_0_1.imag, marker="o", s=10, color="darkred"
)


axes[1][2].plot(unit_circle.real, unit_circle.imag, color="gray", linewidth=0.5)
axes[1][2].scatter(
    eigv_c1_0_1.real, eigv_c1_0_1.imag, marker="o", s=10, color="darkred"
)

axes[2][2].plot(unit_circle.real, unit_circle.imag, color="gray", linewidth=0.5)
axes[2][2].scatter(
    eigv_c1_5_0.real, eigv_c1_5_0.imag, marker="o", s=10, color="darkred"
)

fig.savefig("figures/HW_spectrum_red_lin_op.pdf", pad_inches=3)
