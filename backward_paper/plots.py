# -*- coding: utf-8 -*-
"""
Plotting options
"""
import numpy as np

from matplotlib import use, rc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from config import pkg_path

use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)


def quark_latex_name(name):
    quark_name = name
    if "bar" in name:
        quark_name = r"$\bar{%s}$" % name[0]
    quark_name = quark_name.replace("plus", "+")
    quark_name = quark_name.replace("minus", "-")
    return quark_name


def plot_pdf(log, fig_name, cl=1, plot_pull=False, plot_reldiff=False):
    """
    Plotting routine

    Parameters
    ----------
        log: dict
            log table
        fig_name: str
            Figure name
        cl: int
            confidence level interval ( in units of sigma )
    """
    path = pkg_path / f"{fig_name}.pdf"
    print(f"Writing pdf plots to {path}")

    has_ratio = plot_pull or plot_reldiff

    with PdfPages(path) as pp:
        for name, vals in log.items():

            fig = plt.figure(figsize=(15, 5))
            gs = fig.add_gridspec(3, 2) if has_ratio else fig.add_gridspec(1, 2)

            # compute average and std
            mean = vals.groupby(["x"], axis=0, as_index=False).mean()
            std = vals.groupby(["x"], axis=0, as_index=False).std()

            # plot pdfs with pull
            # col=0 has log x, clo=1 has linear x
            for ncol in [0, 1]:

                ax = (
                    plt.subplot(gs[:-1, ncol])
                    if has_ratio
                    else plt.subplot(gs[:, ncol])
                )
                labels = []
                y_min = 1.0
                for column_name, column_data in mean.iloc[:, 1:].items():
                    ax.plot(mean.x, column_data)
                    ax.fill_between(
                        mean.x,
                        column_data - cl * std[column_name],
                        column_data + cl * std[column_name],
                        alpha=0.2,
                    )
                    labels.append(
                        r"\rm{ %s\ GeV\ }"
                        % (
                            (column_name.replace("_", r"\ ")).replace(
                                ">", r"$\rightarrow$"
                            )
                        )
                    )
                    if np.abs(column_data.min()) < np.abs(y_min):
                        y_min = np.abs(column_data.min())

                ax.set_xlim(mean.x.min(), 1.0)
                ax.legend(labels)
                ax.plot(np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5)

                if ncol == 0:
                    ax.set_ylabel(r"\rm{x %s(x)}" % quark_latex_name(name), fontsize=11)
                    ax.set_xscale("log")
                else:
                    ax.set_yscale("symlog", linthresh=1e-6 if y_min < 1e-6 else y_min)

                if plot_pull:
                    ax_ratio = plt.subplot(gs[-1:, ncol], sharex=ax)
                    for column_name, column_data in mean.iloc[:, 1:].items():
                        ax_ratio.plot(
                            mean.x, column_data.div(std[column_name]).replace(np.nan, 0)
                        )
                    ax_ratio.plot(
                        np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5
                    )
                    ax_ratio.set_xlabel(r"\rm{x}")
                    ax_ratio.set_ylabel(r"\rm{Pull}", fontsize=11)
                elif plot_reldiff:
                    ax_ratio = plt.subplot(gs[-1:, ncol], sharex=ax)
                    ref_data = mean.iloc[:, 1]
                    for column_name, column_data in mean.iloc[:, 2:].items():
                        # TODO: set colors correctly 
                        ax_ratio.plot(
                            mean.x,
                            (column_data - ref_data).div(ref_data).replace(np.nan, 0)
                            * 100,
                            color="#ff7f0e",
                        )
                    ax_ratio.plot(
                        np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5
                    )
                    ax_ratio.set_xlabel(r"\rm{x}")
                    ax_ratio.set_ylabel(r"\rm{\% difference}", fontsize=11)
                    ax_ratio.set_yscale("symlog", linthresh=1e-1)
                    ax_ratio.set_ylim(-100, 100)
                else:
                    ax.set_xlabel(r"\rm{x}")

            plt.tight_layout()
            pp.savefig()
            plt.close(fig)
