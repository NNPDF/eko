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


def plot_pdf(log, fig_name, cl=1):
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

    with PdfPages(path) as pp:
        for name, vals in log.items():

            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            # compute average and std
            mean = vals.groupby(["x"], axis=0, as_index=False).mean()
            std = vals.groupby(["x"], axis=0, as_index=False).std()

            # plot pdfs
            for ax in axs:
                labels = []
                y_min = 1.0
                for column_name, column_data in mean.items():
                    if "x" in column_name or "error" in column_name:
                        continue
                    mean.plot("x", f"{column_name}", ax=ax)

                    ax.fill_between(
                        mean.x,
                        column_data - cl * std[column_name],
                        column_data + cl * std[column_name],
                        alpha=0.2,
                    )
                    labels.append(
                        r"\rm{ %s\ GeV\ }" % (column_name.replace("_", r"\ "))
                    )

                    if np.abs(column_data.min()) < np.abs(y_min):
                        y_min = np.abs(column_data.min())

                quark_name = name
                if "bar" in name:
                    quark_name = r"$\bar{%s}$" % name[0]
                ax.set_xlabel(r"\rm{x}")
                ax.set_xlim(mean.x.min(), 1.0)
                ax.legend(labels)
                ax.plot(np.geomspace(1e-7, 1, 200), np.zeros(200), "k--", alpha=0.5)

            axs[1].set_yscale("symlog", linthresh=1e-7 if y_min < 1e-7 else y_min)
            axs[0].set_xscale("log")
            axs[0].set_ylabel(r"\rm{x %s(x)}" % quark_name, fontsize=11)
            plt.tight_layout()
            pp.savefig()
            plt.close(fig)
