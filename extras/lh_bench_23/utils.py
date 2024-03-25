"""Plotting utils."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FMT_LIST = ["x", "o", "v", "*"]
LHA_LABELS_MAP = {
    "L_m": r"L^- = \bar{d} - \bar{u}",
    "L_p": r"L^+ = 2(\bar{u} + \bar{d})",
    "b_p": "b^+",
    "c_p": "c^+",
    "s_p": "s^+",
}

HERE = pathlib.Path(__file__).parent
plt.style.use(HERE / "plotstyle.mplstyle")


def lha_labels(scheme: str) -> list:
    """PDFs labels in the LHA basis."""
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
    return pdf_labels


def evol_labels(scheme: str) -> list:
    """PDFs labels in the Evolution basis."""
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
    return pdf_labels


def rotate_lha_to_evol(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Rotation from LHA to Evolution basis."""
    if scheme == "FFNS":
        matrix = [
            [1, 1, 0, 0, 0, 0, 0, 0],  # V
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
            [1, 1, 0, 1, -2, 0, 0, 0],  # T8
            [1, 1, 0, 1, 1, -3, 0, 0],  # T15
            [1, 1, 0, 1, 1, 1, -4, 0],  # T24
            [1, 1, 0, 1, 1, 1, 1, 0],  # S
            [0, 0, 0, 0, 0, 0, 0, 1],  # g
        ]
    rotated = (matrix @ df.values.T).T
    return pd.DataFrame(rotated, columns=evol_labels(scheme))


def load_n3lo_tables(
    n3lo_table_dir: pathlib.Path,
    scheme: str,
    sv: str,
    approx: str,
    rotate_to_evol: bool = False,
) -> list:
    """Load the N3LO tables."""
    dfs = []
    for p in n3lo_table_dir.iterdir():
        if scheme not in p.stem:
            continue
        if sv not in p.stem:
            continue

        if approx in p.stem:
            table = pd.read_csv(p, index_col=0)
            table.rename(columns=LHA_LABELS_MAP, inplace=True)
            if rotate_to_evol:
                table = rotate_lha_to_evol(table, scheme)
            dfs.append(table)
    return dfs


def load_nnlo_table(
    table_dir, scheme, sv="central", rotate_to_evol: bool = False
) -> pd.DataFrame:
    """Load the NNLO tables."""
    if scheme == "FFNS":
        tab = 14
    elif scheme == "VFNS":
        tab = 15
    if sv == "central":
        part = 1

    table = pd.read_csv(f"{table_dir}/table{tab}-part{part}.csv", index_col=0)
    table.rename(columns=LHA_LABELS_MAP, inplace=True)
    if rotate_to_evol:
        table = rotate_lha_to_evol(table, scheme)
    return table


def load_msht(
    table_dir: pathlib.Path, scheme: str, approx: str, rotate_to_evol: bool = False
) -> list:
    """Load MSHT files."""

    if scheme != "VFNS":
        raise ValueError(f"{scheme} not provided by MSHT, comment it out")
    APPROX_MAP = {
        "FHMRUVV": "Moch",
        "MSHT": "Posterior",
    }
    fhmruvv_msht_table_dir = table_dir / f"{scheme}_{APPROX_MAP[approx]}_numbers"

    columns = lha_labels(scheme)
    # columns.insert(0,'x')
    # columns.insert(0,'Q')
    dfs = []

    for p in fhmruvv_msht_table_dir.iterdir():
        data = np.loadtxt(p)
        data = pd.DataFrame(data[:, 2:], columns=columns)
        if rotate_to_evol:
            data = rotate_lha_to_evol(data, scheme)
        dfs.append(data)
    return dfs


def compute_n3lo_avg_err(dfs: list) -> tuple:
    """Compute N3LO average and error."""
    df_central = np.mean(dfs, axis=0)
    df_central = pd.DataFrame(df_central, columns=dfs[0].columns)

    # NOTE: here we compute the error as an envelope
    up = np.max(dfs, axis=0)
    dw = np.min(dfs, axis=0)
    df_err = (up - dw) / 2
    df_err = pd.DataFrame(df_err, columns=dfs[0].columns)
    return df_central, df_err


def compute_n3lo_nnlo_diff(n3lo: tuple, nnlo: pd.DataFrame, rel_diff: bool) -> tuple:
    """Compute relative / absolute differece to NNLO."""
    norm = nnlo
    n3lo_central, n3lo_std = n3lo
    if not rel_diff:
        norm = pd.DataFrame(np.full(nnlo.shape, 1), columns=nnlo.columns)
    diff = (n3lo_central - nnlo) / norm
    diff_std = np.abs(n3lo_std / norm)
    return diff, diff_std


def plot_pdfs(
    xgrid: np.array,
    n3lo_dfs: tuple,
    nnlo_df: pd.DataFrame,
    scheme: str,
    use_linx: bool,
    plot_dir: pathlib.Path,
) -> None:
    """Absolute PDFs plots."""

    ncols = 2
    nrows = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))

    xcut = 4 if use_linx else 0
    xgrid = xgrid[xcut:]

    xscale = "linx" if use_linx else "logx"
    plot_name = f"n3lo_bench_{scheme}_{xscale}"
    plot_dir.mkdir(exist_ok=True)

    fig.suptitle(f"{scheme}" + r", $\mu_{\rm f}^2 = 10^4 \ \mbox{GeV}^2$")

    # loop on PDFs
    for i, ax in enumerate(
        axs.reshape(
            8,
        )
    ):
        # loop on n3lo
        for j, (tabs, approx_label) in enumerate(n3lo_dfs):
            central, err = tabs
            ax.errorbar(
                xgrid,
                central.values[xcut:, i],
                yerr=err.values[xcut:, i],
                fmt=FMT_LIST[j],
                label=approx_label,
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
        ax.errorbar(xgrid, nnlo_df.values[xcut:, i], fmt=".", label="NNLO")
        ax.hlines(
            0,
            xgrid.min() - xgrid.min() / 3,
            1,
            linestyles="dotted",
            color="black",
            linewidth=0.5,
        )
        if not use_linx:
            ax.set_xscale("log")
        ax.set_xlabel("$x$")
        ax.set_ylabel(f"${nnlo_df.columns[i]}$")
        ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{plot_name}.pdf")


def plot_diff_to_nnlo(
    xgrid: np.array,
    n3lo_dfs: tuple,
    scheme: str,
    use_linx: bool,
    plot_dir: pathlib.Path,
    rel_dff: bool,
) -> None:
    """Difference w.r.t NNLO PDFs plots."""

    ncols = 2
    nrows = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))

    xcut = 4 if use_linx else 0
    xgrid = xgrid[xcut:]
    xscale = "linx" if use_linx else "logx"

    diff_type = "rel_diff" if rel_dff else "abs_diff"
    plot_name = f"n3lo_bench_{scheme}_{xscale}_{diff_type}"

    diff_type = "Relative" if rel_dff else "Absolute"
    fig.suptitle(
        f"{diff_type} difference to NNLO, {scheme}"
        + r", $\mu_{\rm f}^2 = 10^4 \ \mbox{GeV}^2$"
    )

    for i, ax in enumerate(
        axs.reshape(
            8,
        )
    ):
        # loop on n3lo
        for j, (tabs, approx_label) in enumerate(n3lo_dfs):
            central, err = tabs
            ax.errorbar(
                xgrid,
                central.values[xcut:, i],
                yerr=err.values[xcut:, i],
                fmt=FMT_LIST[j],
                label=approx_label,
                capsize=5,
            )
            # ax.errorbar(
            #     xgrid,
            #     eko_4mom_diff.values[:, i],
            #     yerr=eko_4mom_diff_std.values[:, i],
            #     fmt="x",
            #     label="aN3LO EKO (4 moments)",
            #     capsize=5,
            # )
        ax.hlines(
            0,
            xgrid.min() - xgrid.min() / 3,
            1,
            linestyles="dotted",
            color="black",
            linewidth=0.5,
        )
        if not use_linx:
            ax.set_xscale("log")
        ax.set_xlabel("$x$")
        ax.set_ylabel(f"${central.columns[i]}$")
        ax.set_xlim(xgrid.min() - xgrid.min() / 3, 1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{plot_name}.pdf")
