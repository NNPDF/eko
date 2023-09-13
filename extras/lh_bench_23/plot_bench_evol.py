import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import here, table_dir, xgrid

plot_dir = here / "plots_evol"
plot_dir.mkdir(exist_ok=True)


xgrid = xgrid[4:]

scheme = "FFNS"
sv = "central"
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

plot_name = "lh_n3lo_bench_4mom_linx"


def rotate_to_evol(df):
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
    return (matrix @ df.values[:, 1:].T).T


fhmv_dfs = []
nnpdf_dfs = []
nnpdf_dfs_4mom = []

# load tables
for p in table_dir.iterdir():
    if "FHMV" in p.stem:
        fhmv_dfs.append(rotate_to_evol(pd.read_csv(p)))
    elif "NNPDF-4mom" in p.stem:
        nnpdf_dfs_4mom.append(rotate_to_evol(pd.read_csv(p)))
    elif "NNPDF" in p.stem:
        nnpdf_dfs.append(rotate_to_evol(pd.read_csv(p)))

# load NNLO
if scheme == "FFNS":
    tab = 14
    if sv == "central":
        part = 1
    nnlo_central = pd.read_csv(f"{table_dir}/table{tab}-part{part}.csv")
    nnlo_central = rotate_to_evol(nnlo_central)[4:]


# compute avg and std
nnpdf_central = np.mean(nnpdf_dfs, axis=0)[4:]
nnpdf_std = np.std(nnpdf_dfs, axis=0)[4:]
fhmv_central = np.mean(fhmv_dfs, axis=0)[4:]
fhmv_std = np.std(fhmv_dfs, axis=0)[4:]
nnpdf_4mom_central = np.mean(nnpdf_dfs_4mom, axis=0)[4:]
nnpdf_4mom_std = np.std(nnpdf_dfs_4mom, axis=0)[4:]


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
        nnpdf_central[:, i],
        yerr=nnpdf_std[:, i],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    ax.errorbar(
        xgrid,
        nnpdf_4mom_central[:, i],
        yerr=nnpdf_4mom_std[:, i],
        fmt="x",
        label="aN3LO EKO (4 moments)",
        capsize=5,
    )
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
    # ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{plot_name}.pdf")

# relative diff plots
nnpdf_diff = (nnpdf_central - nnlo_central) / nnlo_central
nnpdf_diff_std = np.abs(nnpdf_std / nnlo_central)
fhmv_diff = (fhmv_central - nnlo_central) / nnlo_central
fhmv_diff_std = np.abs(fhmv_std / nnlo_central)
nnpdf_4mom_diff = (nnpdf_4mom_central - nnlo_central) / nnlo_central
nnpdf_4mom_diff_std = np.abs(nnpdf_4mom_std / nnlo_central)

fig, axs = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle("Relative difference to NNLO, $Q: \\sqrt{2} \\to 100 \\ GeV$")

for i, ax in enumerate(
    axs.reshape(
        8,
    )
):
    ax.errorbar(
        xgrid,
        nnpdf_diff[:, i],
        yerr=nnpdf_diff_std[:, i],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    ax.errorbar(
        xgrid,
        nnpdf_4mom_diff[:, i],
        yerr=nnpdf_4mom_diff_std[:, i],
        fmt="x",
        label="aN3LO EKO (4 moments)",
        capsize=5,
    )
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
    # ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{plot_name}_diff.pdf")
