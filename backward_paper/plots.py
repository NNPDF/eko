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


def plot_pdf(log, pdf_name, q0, cl=1):
    """
    Plotting routine

    Parameters
    ----------
        log: dict
            log table
        pdf_name: str
            PDFs label
        q0: float
            initial scale
        cl: int
            confidence level interval ( in units of sigma )
    """
    path = pkg_path / f"{pdf_name}.pdf"
    print(f"Writing pdf plots to {path}")

    with PdfPages(path) as pp:
        for name, vals in log.items():

            fig = plt.figure(figsize=(15, 5))
            plt.title(name)
            # compute average and std
            mean = vals.groupby(["x"], axis=0, as_index=False).mean()
            std = vals.groupby(["x"], axis=0, as_index=False).std()

            # plot evoluted result
            ax = mean.plot("x", "eko")
            plt.fill_between(
                std.x, mean.eko - cl * std.eko, mean.eko + cl * std.eko, alpha=0.2
            )
            # plot initial pdf
            mean.plot("x", "inputpdf", ax=ax)
            plt.fill_between(
                std.x,
                mean.inputpdf - cl * std.inputpdf,
                mean.inputpdf + cl * std.inputpdf,
                alpha=0.2,
            )

            plt.xscale("log")
            quark_name = name
            if "bar" in name:
                quark_name = r"$\bar{%s}$" % name[0]
            plt.ylabel(r"\rm{x %s(x)}" % quark_name, fontsize=11)
            plt.xlabel(r"\rm{x}")
            plt.xlim(1e-2, 1.0)
            plt.ylim(-0.04, 0.04)
            ax.legend(
                [
                    r"\rm{ %s\ @\ %s\ GeV\ }"
                    % (pdf_name.replace("_", r"\ "), np.round(np.sqrt(mean.Q2[0]), 2)),
                    r"\rm{ %s\ @\ %s\ GeV\ }"
                    % (pdf_name.replace("_", r"\ "), np.round(q0, 2)),
                ]
            )
            plt.plot(np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5)
            plt.tight_layout()
            pp.savefig()
            plt.close(fig)
