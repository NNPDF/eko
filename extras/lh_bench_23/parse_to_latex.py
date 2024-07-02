from cfg import here, table_dir, xgrid
from utils import compute_n3lo_avg_err, load_n3lo_tables

n3lo_table_dir = table_dir

latex_tab = here / "latex_tab"
latex_tab.mkdir(exist_ok=True)

SVS = ["central", "up", "down"]

MIDRULE1 = r"""
\hline \hline
\multicolumn{9}{||c||}{} \\[-3mm]
\multicolumn{9}{||c||}{"""

MIDRULE2 = r"""} \\
\multicolumn{9}{||c||}{} \\[-0.3cm]
\hline \hline
 & & & & & & & \\[-0.3cm]
"""

BOTTOMRULE = r"""
\hline \hline
\end{tabular}
\end{center}
\end{table}
"""

VFNS_LABELS = r"""
    \multicolumn{1}{c|} {$xu_v$} &
    \multicolumn{1}{c|} {$xd_v$} &
    \multicolumn{1}{c|} {$xL_-$} &
    \multicolumn{1}{c|} {$xL_+$} &
    \multicolumn{1}{c|} {$xs_+$} &
    \multicolumn{1}{c|} {$xc_+$} &
    \multicolumn{1}{c|} {$xb_+$} &
    \multicolumn{1}{c||}{$xg$} \\[0.5mm]
    """

FFNS_LABELS = r"""
    \multicolumn{1}{c|} {$xu_v$} &
    \multicolumn{1}{c|} {$xd_v$} &
    \multicolumn{1}{c|} {$xL_-$} &
    \multicolumn{1}{c|} {$xL_+$} &
    \multicolumn{1}{c|} {$xs_v$} &
    \multicolumn{1}{c|} {$xs_+$} &
    \multicolumn{1}{c|} {$xc_+$} &
    \multicolumn{1}{c||}{$xg$}
    """


def insert_headrule(scheme, approx, caption):
    """Insert the middle rule."""
    label = r"\label{tab:" + f"n3lo_{scheme.lower()}_{approx.lower()}" + "}"
    scheme_label = (
        r", $\, N_{\rm f} = 3\ldots 5\,$,"
        if scheme == "VFNS"
        else r"$\, N_{\rm f} = 4$,"
    )
    HEADRULE = (
        r"""
    \begin{table}[htp]
    \caption{"""
        + caption
        + r"""}
    """
        + label
        + r"""
    \begin{center}
    \vspace{5mm}
    \begin{tabular}{||c||r|r|r|r|r|r|r|r||}
    \hline \hline
    \multicolumn{9}{||c||}{} \\[-3mm]
    \multicolumn{9}{||c||}{"""
        # + r"""aN$^3$LO, """
        + approx
        + scheme_label
        + r"""$\,\mu_{\rm f}^2 = 10^4 \mbox{ GeV}^2$} \\
    \multicolumn{9}{||c||}{} \\[-0.3cm]
    \hline \hline
    \multicolumn{9}{||c||}{} \\[-3mm]
    \multicolumn{1}{||c||}{$x$} &
    """
    )
    HEADRULE += VFNS_LABELS if scheme == "VFNS" else FFNS_LABELS
    HEADRULE += r"""\\[0.5mm]"""
    return HEADRULE


def insert_midrule(sv):
    """Insert the middle rule."""
    # TODO: is this mapping correct or the other way round ??
    # xif2 = 2 -> up
    # xif2 = 1/2 -> down
    label = {
        "central": r"$\mu_{\rm r}^2 = \ \mu_{\rm f}^2$",
        "down": r"$\mu_{\rm r}^2 = 0.5 \ \mu_{\rm f}^2$",
        "up": r"$\mu_{\rm r}^2 = 2 \ \mu_{\rm f}^2$",
    }
    return MIDRULE1 + label[sv] + MIDRULE2


def format_float(values):
    """Clean float format."""
    values = values.replace("0000", "0")
    values = values.replace("e-0", "$^{-")
    values = values.replace("e-10", "$^{-10")
    values = values.replace("e+0", "$^{+")
    values = values.replace("&", "}$ &")
    values = values.replace(r"\\", r"}$ \\")
    return values


def dump_table(scheme: str, approx: str, caption: str):
    """Write a nice latex table."""
    final_tab = insert_headrule(scheme, approx.replace("EKO", "NNPDF"), caption)
    # loop on scales
    for sv in SVS:
        # load tables
        dfs = load_n3lo_tables(n3lo_table_dir, scheme, sv=sv, approx=approx)

        central, err = compute_n3lo_avg_err(dfs)

        central.insert(0, "x", xgrid)
        values = central.to_latex(float_format="{:.4e}".format, index=False)
        values = "".join(e for e in values.split("\n")[4:-3])
        final_tab += insert_midrule(sv) + format_float(values)

    final_tab += BOTTOMRULE

    # write
    with open(
        latex_tab / f"table-{scheme}-{approx.replace('EKO', 'NNPDF')}.tex",
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(final_tab)


if __name__ == "__main__":
    approx = "FHMRUVV"
    scheme = "FFNS"
    caption = r"""
        Results for the FFNS aN$^3$LO evolution
        for the initial conditions and the input parton distributions
        given in Sec.~\ref{sec:toy_pdf},
        with the FHMRUVV splitting functions approximation and the NNPDF code.
    """
    dump_table(scheme, approx, caption)

    approx = "FHMRUVV"
    scheme = "VFNS"
    caption = r"""
        Results for the VFNS aN$^3$LO evolution
        for the initial conditions and the input parton distributions
        given in Sec.~\ref{sec:toy_pdf},
        with the FHMRUVV splitting functions approximation and the NNPDF code.
    """
    dump_table(scheme, approx, caption)

    approx = "EKO"
    scheme = "FFNS"
    caption = r"""
        Results for the FFNS aN$^3$LO evolution
        for the initial conditions and the input parton distributions
        given in Sec.~\ref{sec:toy_pdf},
        with the NNPDF splitting functions approximation.
    """
    dump_table(scheme, approx, caption)

    approx = "EKO"
    scheme = "VFNS"
    caption = r"""
        Results for the VFNS aN$^3$LO evolution
        for the initial conditions and the input parton distributions
        given in Sec.~\ref{sec:toy_pdf},
        with the NNPDF splitting functions approximation.
    """
    dump_table(scheme, approx, caption)
