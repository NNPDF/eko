import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cfg import table_dir, xgrid
from utils import HERE, lha_labels

from eko.io.types import EvolutionMethod

plt.style.use(HERE / "plotstyle.mplstyle")
plot_dir = HERE / "plots_trn_exa"

PTOS = {1: "NLO", 2: "NNLO", 3: "N$^3$LO"}

COLUMNS_TO_KEEP = ["L_m", "L_p", "g"]


def load_table(method):
    """Load tables."""
    dfs = {}
    for pto in PTOS:
        with open(table_dir / f"table_FFNS-{pto}_{method}.csv", encoding="utf-8") as f:
            dfs[pto] = pd.read_csv(f, index_col=0)
    return dfs


def plot_diff(xgrid, dfs_trn, dfs_exa):
    cut_smallx = 0
    cut_largex = -1
    xgrid = xgrid[cut_smallx:cut_largex]

    plot_dir.mkdir(exist_ok=True)

    # loop on PDFs
    for column in COLUMNS_TO_KEEP:
        _, ax = plt.subplots(1, 1, figsize=(1 * 5, 1 * 3.5))
        j = np.where(dfs_trn[1].columns == column)[0][0]

        # loop on ptos
        for pto, pto_label in PTOS.items():
            diff = (dfs_trn[pto] - dfs_exa[pto]) / dfs_trn[pto] * 100

            ax.plot(xgrid, diff.values[cut_smallx:cut_largex, j], label=pto_label)
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
        ax.set_ylabel(f'${lha_labels("FFNS")[j]}$')
        ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/diff_trn_exa_{column}.pdf")


if __name__ == "__main__":
    dfs_trn = load_table(EvolutionMethod.TRUNCATED.value)
    dfs_exa = load_table(EvolutionMethod.ITERATE_EXACT.value)
    plot_diff(xgrid, dfs_trn, dfs_exa)
