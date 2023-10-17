import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import here, table_dir, xgrid

plot_dir = here / "plots"
plot_dir.mkdir(exist_ok=True)

USE_LINX = False
REL_DIFF = False
scheme = "VFNS"
sv = "central"

if scheme == "FFNS":
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
elif scheme == "VFNS":
    pdf_labels = [
        "u_v",
        "d_v",
        r"L^- = \bar{d} - \bar{u}",
        r"L^+ = 2(\bar{u} + \bar{d})",
        "s^+",
        "c^+",
        "b^+",
        "g",
    ]
XCUT = 4 if USE_LINX else 0
xscale = "linx" if USE_LINX else "logx"
xgrid = xgrid[XCUT:]
plot_name = f"lh_n3lo_bench_{scheme}_{xscale}"

fhmv_dfs = []
eko_dfs = []
# eko_dfs_4mom = []

# load tables
n3lo_table_dir = table_dir / scheme
for p in n3lo_table_dir.iterdir():
    if "FHMV" in p.stem:
        fhmv_dfs.append(pd.read_csv(p))
    # elif "EKO-4mom" in p.stem:
    #     eko_dfs_4mom.append(pd.read_csv(p))
    elif "EKO" in p.stem:
        eko_dfs.append(pd.read_csv(p))

# load NNLO
if scheme == "FFNS":
    tab = 14
elif scheme == "VFNS":
    tab = 15
if sv == "central":
    part = 1

nnlo_central = pd.read_csv(f"{table_dir}/table{tab}-part{part}.csv")
nnlo_central = nnlo_central[XCUT:]

# compute avg and std
eko_central = np.mean(eko_dfs, axis=0)[XCUT:]
eko_std = np.std(eko_dfs, axis=0)[XCUT:]
fhmv_central = np.mean(fhmv_dfs, axis=0)[XCUT:]
fhmv_std = np.std(fhmv_dfs, axis=0)[XCUT:]
# eko_4mom_central = np.mean(eko_dfs_4mom, axis=0)[XCUT:]
# eko_4mom_std = np.std(eko_dfs_4mom, axis=0)[XCUT:]


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
        eko_central[:, i + 1],
        yerr=eko_std[:, i + 1],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    # ax.errorbar(
    #     xgrid,
    #     eko_4mom_central[:, i + 1],
    #     yerr=eko_4mom_std[:, i + 1],
    #     fmt="x",
    #     label="aN3LO EKO (4 moments)",
    #     capsize=5,
    # )
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
    if not USE_LINX:
        ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{plot_name}.pdf")

# relative, absolute diff plots
norm = nnlo_central
if not REL_DIFF:
    norm = pd.DataFrame(np.full(nnlo_central.shape, 1), columns=nnlo_central.columns)
eko_diff = (eko_central - nnlo_central) / norm
eko_diff_std = np.abs(eko_std / norm)
fhmv_diff = (fhmv_central - nnlo_central) / norm
fhmv_diff_std = np.abs(fhmv_std / norm)
# eko_4mom_diff = (eko_4mom_central - nnlo_central) / norm
# eko_4mom_diff_std = np.abs(eko_4mom_std / norm)

fig, axs = plt.subplots(2, 4, figsize=(15, 7))
if REL_DIFF:
    fig.suptitle("Relative difference to NNLO, $Q: \\sqrt{2} \\to 100 \\ GeV$")
else:
    fig.suptitle("Absolute difference to NNLO, $Q: \\sqrt{2} \\to 100 \\ GeV$")

for i, ax in enumerate(
    axs.reshape(
        8,
    )
):
    ax.errorbar(
        xgrid,
        eko_diff.values[:, i + 1],
        yerr=eko_diff_std.values[:, i + 1],
        fmt="x",
        label="aN3LO EKO",
        capsize=5,
    )
    # ax.errorbar(
    #     xgrid,
    #     eko_4mom_diff.values[:, i + 1],
    #     yerr=eko_4mom_diff_std.values[:, i + 1],
    #     fmt="x",
    #     label="aN3LO EKO (4 moments)",
    #     capsize=5,
    # )
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
    if not USE_LINX:
        ax.set_xscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${pdf_labels[i]}$")
    ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

plt.legend()
plt.tight_layout()
if REL_DIFF:
    plt.savefig(f"{plot_dir}/{plot_name}_diff.pdf")
else:
    plt.savefig(f"{plot_dir}/{plot_name}_abs_diff.pdf")
