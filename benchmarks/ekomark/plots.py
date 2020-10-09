# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_dist(x, y, yerr, yref, title=None, oMx_min=1e-2, oMx_max=0.5):
    """
    Compare two distributions.

    Generates a plot with 3 areas:

    - small x, i.e. log(x) as abscissa
    - linear x, i.e. with id(x) as abscissa
    - large x, i.e. with log(1-x) as abscissa


    Parameters
    ----------
        x : numpy.ndarray
            list of abscisses
        y : numpy.ndarray
            computed list of ordinates
        yerr : numpy.ndarray
            list of ordinate errors
        yref : numpy.ndarray
            reference list of ordinates

    Other Parameters
    ----------------
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
    ax1.set_xscale("log", nonpositive="clip")
    ax1.set_yscale("log", nonpositive="clip")
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.loglog(x, yref, "x")
    ax1b = plt.subplot(2, 3, 4, sharex=ax1)
    ax1b.set_xscale("log", nonpositive="clip")
    ax1b.set_yscale("log", nonpositive="clip")
    plt.errorbar(x, np.abs((y - yref) / yref), yerr=np.abs(yerr / yref), fmt="s")
    plt.xlabel("x")
    # linear x
    ax2 = plt.subplot(2, 3, 2)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.title("linear x")
    plt.errorbar(x, y, yerr=yerr, fmt="o")
    plt.plot(x, yref, "x")
    ax2b = plt.subplot(2, 3, 5, sharex=ax2)
    ax2b.set_yscale("log", nonpositive="clip")
    plt.errorbar(x, np.abs((y - yref) / yref), yerr=np.abs(yerr / yref), fmt="s")
    plt.xlabel("x")
    # large x
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_xscale("log", nonpositive="clip")
    ax3.set_yscale("log", nonpositive="clip")
    oMx = 1.0 - x
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_xlim(oMx_min, oMx_max)
    plt.title("large x, i.e. small (1-x)")
    plt.errorbar(oMx, y, yerr=yerr, fmt="o")
    plt.loglog(oMx, yref, "x")
    ax3b = plt.subplot(2, 3, 6, sharex=ax3)
    ax3b.set_xscale("log", nonpositive="clip")
    ax3b.set_yscale("log", nonpositive="clip")
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
    if np.max(op) <= 0.0:
        return fig

    # TODO fix File "/usr/lib/python3/dist-packages/matplotlib/colors.py", line 1181, in _check_vmin_vmax
    # raise ValueError("minvalue must be positive")
    # ValueError: minvalue must be positive
    # import pdb; pdb.set_trace()

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
    err_to_val = np.abs(np.array(op_err) / np.array(op))
    im = plt.imshow(err_to_val, norm=LogNorm(), aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)
    return fig
