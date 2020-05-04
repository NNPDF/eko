# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages


def plot_dist(x, y, yerr, yref, title=None, oMx_min=1e-2, oMx_max=0.5):
    """
        Compare two distributions.

        Generates a plot with 3 areas:
        - small x, i.e. log(x) as abscissa
        - linear x, i.e. with id(x) as abscissa
        - large x, i.e. with log(1-x) as abscissa

        Parameters
        ----------
            x : array
                list of abscisses
            y : array
                computed list of ordinates
            yerr : array
                list of ordinate errors
            yref : array
                reference list of ordinates

        Additional Parameters
        ---------------------
            title : string, optional
                additional overall title
            oMx_min : float
                maximum value for the large x region, i.e. 1-x > 1 - `oMx_min`
            oMx_max : float
                minimum value for the large x region, i.e. 1 - `oMx_max` > 1-x

        Returns
        -------
            fig : matplotlib.pyplot.figure
                created figure
    """
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(hspace=0.05)
    if title is not None:
        fig.suptitle(title)
    # small x
    ax1 = plt.subplot(2, 3, 1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.title("small x")
    ax1.set_xscale("log", nonposx="clip")
    ax1.set_yscale("log", nonposy="clip")
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.loglog(x, yref, "x")
    ax1b = plt.subplot(2, 3, 4, sharex=ax1)
    ax1b.set_xscale("log", nonposx="clip")
    ax1b.set_yscale("log", nonposy="clip")
    plt.errorbar(x, np.abs((y - yref) / yref), yerr=np.abs(yerr / yref), fmt="s")
    plt.xlabel("x")
    # linear x
    ax2 = plt.subplot(2, 3, 2)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.title("linear x")
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.plot(x, yref, "x")
    ax2b = plt.subplot(2, 3, 5, sharex=ax2)
    ax2b.set_yscale("log", nonposy="clip")
    plt.errorbar(x, np.abs((y - yref) / yref), yerr=np.abs(yerr / yref), fmt="s")
    plt.xlabel("x")
    # large x
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_xscale("log", nonposx="clip")
    ax3.set_yscale("log", nonposy="clip")
    oMx = 1.0 - x
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_xlim(oMx_min, oMx_max)
    plt.title("large x, i.e. small (1-x)")
    plt.errorbar(oMx, y, yerr=yerr, fmt="o")
    plt.loglog(oMx, yref, "x")
    ax3b = plt.subplot(2, 3, 6, sharex=ax3)
    ax3b.set_xscale("log", nonposx="clip")
    ax3b.set_yscale("log", nonposy="clip")
    plt.errorbar(oMx, np.abs((y - yref) / yref), yerr=np.abs(yerr / yref), fmt="s")
    plt.xlabel("1-x")
    return fig


def plot_operator(ret, var_name, log_operator=True, abs_operator=False):
    """
        Plot a single operator as heat map.

        Parameters
        ----------
            ret : dict
                DGLAP result
            var_name : string
                operator name
            log_operator : bool, optional
                plot on logarithmic scale
            abs_operator : bool, optional
                plot absolute value (instead of true value)

        Returns
        -------
            fig : matplotlib.pyplot.figure
                created figure
    """
    # get
    op = ret["operators"][var_name]
    op_err = ret["operator_errors"][var_name]

    fig = plt.figure(figsize=(25, 5))
    fig.suptitle(var_name)

    # empty?
    if np.max(op) == 0.0:
        return fig

    ax = plt.subplot(1, 3, 1)
    if abs_operator:
        plt.title("|operator|")
    else:
        plt.title("operator")
    norm = LogNorm() if log_operator else None
    if abs_operator:
        op = np.abs(op)
    im = plt.imshow(op, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)

    ax = plt.subplot(1, 3, 2)
    plt.title("operator_error")
    im = plt.imshow(op_err, norm=LogNorm(), aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)

    ax = plt.subplot(1, 3, 3)
    plt.title("|error/value|")
    err_to_val = np.abs(op_err / op)
    im = plt.imshow(err_to_val, norm=LogNorm(), aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)
    return fig

# https://stackoverflow.com/a/7205107
# from functools import reduce
# reduce(merge, [dict1, dict2, dict3...])
def merge_dicts(a: dict, b: dict, path=None):
    """
        Merges b into a.

        Parameters
        ----------
            a : dict
                target dictionary (modified)
            b : dict
                update
            path : array
                recursion track

        Returns
        -------
            a : dict
                updated dictionary
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def save_all_operators_to_pdf(ret, path):
    """
        Output all operator heatmaps to PDF.

        Parameters
        ----------
            ret : dict
                (single) operator matrices
            path : string
                target file name
    """
    pp = PdfPages(path)
    # NS
    #fig = plot_operator(ret, "V.V", log_operator=False)
    #pp.savefig()
    #plt.close(fig)
    #fig = plot_operator(ret, "V.V", abs_operator=True)
    #pp.savefig()
    #plt.close(fig)
    fig = plot_operator(ret, "V.V")
    pp.savefig()
    plt.close(fig)
    # Singlet
    fig = plot_operator(ret, "S.S")
    pp.savefig()
    plt.close(fig)
    fig = plot_operator(ret, "S.g")
    pp.savefig()
    plt.close(fig)
    fig = plot_operator(ret, "g.S")
    pp.savefig()
    plt.close(fig)
    fig = plot_operator(ret, "g.g")
    pp.savefig()
    plt.close(fig)
    # close
    pp.close()
