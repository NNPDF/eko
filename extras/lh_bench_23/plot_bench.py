import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import here, table_dir, xgrid

plot_dir = here / "plots"
plot_dir.mkdir(exist_ok=True)


scheme = "FFNS"
sv = "central"
pdf_labels = [
    "u_v",
    "d_v",
    r"L^- = \bar{d} - \bar{u}",
    r"L^+ = 2(\bar{u} + \bar{d})",
    "s_v",
    "s^+",
    "c^+",
    "g",
]

fhmv_dfs = []
nnpdf_dfs = []

# load tables
for p in table_dir.iterdir():
    if "FHMV" in p.stem:
        fhmv_dfs.append(pd.read_csv(p))
    if "NNPDF" in p.stem:
        nnpdf_dfs.append(pd.read_csv(p))

# load NNLO
if scheme == "FFNS":
    tab = 14
    if sv == "central":
        part = 1
    nnlo_central = pd.read_csv(f"{table_dir}/table{tab}-part{part}.csv")

# compute avg and std
nnpdf_central = np.mean(nnpdf_dfs, axis=0)
nnpdf_std = np.std(nnpdf_dfs, axis=0)
fhmv_central = np.mean(fhmv_dfs, axis=0)
fhmv_std = np.std(fhmv_dfs, axis=0)


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
        nnpdf_central[:, i + 1],
        yerr=nnpdf_std[:, i + 1],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    ax.errorbar(
        xgrid,
        fhmv_central[:, i + 1],
        yerr=fhmv_std[:, i + 1],
        fmt="o",
        label="aN3LO FHMV",
        capsize=5,
    )
    ax.errorbar(xgrid, nnlo_central.values[:, i + 1], fmt=".", label="NNLO")
    ax.hlines(
        0,
        xgrid.min() - xgrid.min() / 3,
        1,
        linestyles="dotted",
        color="black",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/lh_n3lo_bench.pdf")

# relative diff plots
nnpdf_diff = (nnpdf_central - nnlo_central) / nnlo_central
nnpdf_diff_std = np.abs(nnpdf_std / nnlo_central)
fhmv_diff = (fhmv_central - nnlo_central) / nnlo_central
fhmv_diff_std = np.abs(fhmv_std / nnlo_central)


fig, axs = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle("Relative difference to NNLO, $Q: \\sqrt{2} \\to 100 \\ GeV$")

for i, ax in enumerate(
    axs.reshape(
        8,
    )
):
    ax.errorbar(
        xgrid,
        nnpdf_diff.values[:, i + 1],
        yerr=nnpdf_diff_std.values[:, i + 1],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    ax.errorbar(
        xgrid,
        fhmv_diff.values[:, i + 1],
        yerr=fhmv_diff_std.values[:, i + 1],
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
    ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/lh_n3lo_bench_diff.pdf")
