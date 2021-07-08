# -*- coding: utf-8 -*-
"""
Plotting options
"""
import pathlib
import numpy as np

from matplotlib import use, rc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


use("PDF")
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

pkg_path = pathlib.Path(__file__).absolute().parents[0]


def plot_pdf(log, fig_name, cl=1, logscale=True):
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
        logscale: bool
            plot with x axis in log scale
    """
    path = pkg_path / f"{fig_name}.pdf"
    print(f"Writing pdf plots to {path}")

    with PdfPages(path) as pp:
        for name, vals in log.items():

            fig = plt.figure(figsize=(15, 5))
            plt.title(name)
            # compute average and std
            mean = vals.groupby(["x"], axis=0, as_index=False).mean()
            std = vals.groupby(["x"], axis=0, as_index=False).std()

            # plot pdfs
            labels = []
            for idx, (column_name, column_data) in enumerate(mean.items()):
                if "x" in column_name or "error" in column_name:
                    continue
                if idx == 1:
                    ax = mean.plot("x", f"{column_name}")
                else:
                    mean.plot("x", f"{column_name}", ax=ax)

                plt.fill_between(
                    mean.x,
                    column_data - cl * std[column_name],
                    column_data + cl * std[column_name],
                    alpha=0.2,
                )
                labels.append(r"\rm{ %s\ GeV\ }" % (column_name.replace("_", r"\ ")))

            if logscale:
                plt.xscale("log")
            quark_name = name
            if "bar" in name:
                quark_name = r"$\bar{%s}$" % name[0]
            plt.ylabel(r"\rm{x %s(x)}" % quark_name, fontsize=11)
            plt.xlabel(r"\rm{x}")
            plt.xlim(std.x.min(), 1.0)
            # plt.ylim(-0.04, 0.04)
            ax.legend(labels)
            plt.plot(np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5)
            plt.tight_layout()
            pp.savefig()
            plt.close(fig)
