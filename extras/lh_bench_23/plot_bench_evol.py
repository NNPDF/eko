import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import here, table_dir, xgrid

plot_dir = here / "plots_evol"
plot_dir.mkdir(exist_ok=True)

USE_LINX = False

scheme = "VFNS"
sv = "central"

if scheme == "FFNS":
    pdf_labels = [
        "V",
        "V_3",
        "V_8",
        "T_3",
        "T_8",
        "T_{15}",
        r"\Sigma",
        "g",
    ]
elif scheme == "VFNS":
    pdf_labels = [
        "V",
        "V_3",
        "T_3",
        "T_8",
        "T_{15}",
        "T_{24}",
        r"\Sigma",
        "g",
    ]

XCUT = 4 if USE_LINX else 0
xscale = "linx" if USE_LINX else "logx"
xgrid = xgrid[XCUT:]
plot_name = f"lh_n3lo_bench_{scheme}_{xscale}"


def rotate_to_evol(df, scheme):
    if scheme == "FFNS":
        matrix = [
            [1, 1, 0, 0, 1, 0, 0, 0],  # V
            [1, -1, 0, 0, 0, 0, 0, 0],  # V3
            [1, 1, 0, 0, -2, 0, 0, 0],  # V8
            [1, -1, -2, 0, 0, 0, 0, 0],  # T3
            [1, 1, 0, 1, 0, -2, 0, 0],  # T8
            [1, 1, 0, 1, 0, 1, -3, 0],  # T15
            [1, 1, 0, 1, 0, 1, 1, 0],  # S
            [0, 0, 0, 0, 0, 0, 0, 1],  # g
        ]
    elif scheme == "VFNS":
        matrix = [
            [1, 1, 0, 0, 0, 0, 0, 0],  # V, V8
            [1, -1, 0, 0, 0, 0, 0, 0],  # V3
            [1, -1, -2, 0, 0, 0, 0, 0],  # T3
            [1, 1, 0, 2, -2, 0, 0, 0],  # T8
            [1, 1, 0, 2, 1, -3, 0, 0],  # T15
            [1, 1, 0, 2, 1, 1, -4, 0],  # T24
            [1, 1, 0, 2, 1, 1, 1, 0],  # S
            [0, 0, 0, 0, 0, 0, 0, 1],  # g
        ]

    return (matrix @ df.values[:, 1:].T).T


fhmv_dfs = []
eko_dfs = []
eko_dfs_4mom = []

# load tables
for p in table_dir.iterdir():
    if "FHMV" in p.stem:
        fhmv_dfs.append(rotate_to_evol(pd.read_csv(p), scheme))
    elif "EKO-4mom" in p.stem:
        eko_dfs_4mom.append(rotate_to_evol(pd.read_csv(p), scheme))
    elif "EKO" in p.stem:
        eko_dfs.append(rotate_to_evol(pd.read_csv(p), scheme))

# load NNLO
if scheme == "FFNS":
    tab = 14
elif scheme == "VFNS":
    tab = 15
if sv == "central":
    part = 1
nnlo_central = pd.read_csv(f"{table_dir}/table{tab}-part{part}.csv")
nnlo_central = rotate_to_evol(nnlo_central, scheme)[XCUT:]

# compute avg and std
eko_central = np.mean(eko_dfs, axis=0)[XCUT:]
eko_std = np.std(eko_dfs, axis=0)[XCUT:]
fhmv_central = np.mean(fhmv_dfs, axis=0)[XCUT:]
fhmv_std = np.std(fhmv_dfs, axis=0)[XCUT:]
eko_4mom_central = np.mean(eko_dfs_4mom, axis=0)[XCUT:]
eko_4mom_std = np.std(eko_dfs_4mom, axis=0)[XCUT:]


# absolute plots
fig, axs = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle("$Q: \\sqrt{2} \\to 100 \\ GeV$")
for i, ax in enumerate(
    axs.reshape(
        8,
    )
):
    ax.errorbar(
        xgrid,
        eko_central[:, i],
        yerr=eko_std[:, i],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    # ax.errorbar(
    #     xgrid,
    #     eko_4mom_central[:, i],
    #     yerr=eko_4mom_std[:, i],
    #     fmt="x",
    #     label="aN3LO EKO (4 moments)",
    #     capsize=5,
    # )
    ax.errorbar(
        xgrid,
        fhmv_central[:, i],
        yerr=fhmv_std[:, i],
        fmt="o",
        label="aN3LO FHMV",
        capsize=5,
    )
    ax.errorbar(xgrid, nnlo_central[:, i], fmt=".", label="NNLO")
    ax.hlines(
        0,
        xgrid.min() - xgrid.min() / 3,
        1,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    if not USE_LINX:
        ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{plot_name}.pdf")

# relative diff plots
eko_diff = (eko_central - nnlo_central) / nnlo_central
eko_diff_std = np.abs(eko_std / nnlo_central)
fhmv_diff = (fhmv_central - nnlo_central) / nnlo_central
fhmv_diff_std = np.abs(fhmv_std / nnlo_central)
eko_4mom_diff = (eko_4mom_central - nnlo_central) / nnlo_central
eko_4mom_diff_std = np.abs(eko_4mom_std / nnlo_central)

fig, axs = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle("Relative difference to NNLO, $Q: \\sqrt{2} \\to 100 \\ GeV$")

for i, ax in enumerate(
    axs.reshape(
        8,
    )
):
    ax.errorbar(
        xgrid,
        eko_diff[:, i],
        yerr=eko_diff_std[:, i],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    # ax.errorbar(
    #     xgrid,
    #     eko_4mom_diff[:, i],
    #     yerr=eko_4mom_diff_std[:, i],
    #     fmt="x",
    #     label="aN3LO EKO (4 moments)",
    #     capsize=5,
    # )
    ax.errorbar(
        xgrid,
        fhmv_diff[:, i],
        yerr=fhmv_diff_std[:, i],
        fmt="o",
        label="aN3LO FHMV",
        capsize=5,
    )
    ax.hlines(
        0,
        xgrid.min() - xgrid.min() / 3,
        1,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    if not USE_LINX:
        ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{plot_name}_diff.pdf")
