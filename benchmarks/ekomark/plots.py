# -*- coding: utf-8 -*-
import io
import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages


def input_figure(theory, ops, pdf_name=None):
    """
    Pretty-prints the setup to a figure

    Parameters
    ----------
        theory : dict
            theory card
        ops : dict
            operator card
        pdf_name : str
            PDF name

    Returns
    -------
        firstPage : matplotlib.pyplot.Figure
            figure
    """
    firstPage = plt.figure(figsize=(25, 20))
    # theory
    firstPage.text(0.05, 0.97, "Theory:", size=20, ha="left", va="top")
    str_stream = io.StringIO()
    th_copy = theory.copy()
    th_copy.update({"hash": theory["hash"].hex()[:7]})
    pprint.pprint(th_copy, stream=str_stream, width=50)
    firstPage.text(0.05, 0.92, str_stream.getvalue(), size=14, ha="left", va="top")
    # operators
    firstPage.text(0.55, 0.87, "Operators:", size=20, ha="left", va="top")
    str_stream = io.StringIO()
    ops_copy = ops.copy()
    ops_copy.update({"hash": ops["hash"].hex()[:7]})
    pprint.pprint(ops_copy, stream=str_stream, width=50)
    firstPage.text(0.55, 0.82, str_stream.getvalue(), size=14, ha="left", va="top")
    # pdf
    if pdf_name is not None:
        firstPage.text(0.55, 0.97, "source pdf:", size=20, ha="left", va="top")
        firstPage.text(0.55, 0.92, pdf_name, size=14, ha="left", va="top")
        firstPage.tight_layout()
    return firstPage


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
    np.seterr(divide="ignore", invalid="ignore")
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
    x = np.array(x)
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


def plot_operator(var_name, op, op_err, log_operator=False, abs_operator=False):
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
    # op = ret["operators"][var_name]
    # op_err = ret["operator_errors"][var_name]

    fig = plt.figure(figsize=(25, 5))
    fig.suptitle(var_name)

    # empty?
    if np.max(op) <= 0.0:
        return fig

    # TODO fix File "/usr/lib/python3/dist-packages/matplotlib/colors.py",
    # line 1181, in _check_vmin_vmax
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

    im = plt.imshow(op_err, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)

    ax = plt.subplot(1, 3, 3)
    plt.title("|error/value|")
    err_to_val = np.abs(np.array(op_err) / np.array(op))
    im = plt.imshow(err_to_val, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.034, pad=0.04)
    return fig


def save_operators_to_pdf(path, theory, ops, me, skip_pdfs):
    """
        Output all operator heatmaps to PDF.

    Parameters
    ----------
        path : str
            path to the plot
        theory : dict
            theory card
        ops : dict
            operator card
        me : eko.output.Output
            DGLAP result
        skip_pdfs : list
            PDF to skip
    """

    ops_names = list(me["pids"])
    ops_id = f"o{ops['hash'].hex()[:6]}_t{theory['hash'].hex()[:6]}"
    path = f"{path}/{ops_id}.pdf"
    print(f"Plotting operators plots to {path}")

    with PdfPages(path) as pp:
        # print setup
        firstPage = input_figure(theory, ops)
        pp.savefig()
        plt.close(firstPage)

        # plot the operators
        # it's necessary to reshuffle the eko output
        for q2 in me["Q2grid"]:
            results = me["Q2grid"][q2]["operators"]
            errors = me["Q2grid"][q2]["operator_errors"]

            # loop on pids
            for label_out, res, res_err in zip(ops_names, results, errors):
                if label_out in skip_pdfs:
                    continue
                new_op = {}
                new_op_err = {}
                # loop on xgrid point
                for j in range(len(me["interpolation_xgrid"])):
                    # loop on pid in
                    for label_in, val, val_err in zip(ops_names, res[j], res_err[j]):
                        if label_in in skip_pdfs:
                            continue
                        if label_in not in new_op.keys():
                            new_op[label_in] = []
                            new_op_err[label_in] = []
                        new_op[label_in].append(val)
                        new_op_err[label_in].append(val_err)

                for label_in in ops_names:
                    if label_in in skip_pdfs:
                        continue
                    try:
                        fig = plot_operator(
                            f"Operator ({label_in};{label_out}) µ_F^2 = {q2} GeV^2",
                            new_op[label_in],
                            new_op_err[label_in],
                        )
                        pp.savefig()
                    finally:
                        if fig:
                            plt.close(fig)
