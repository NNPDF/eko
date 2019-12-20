# -*- coding: utf-8 -*-
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import eko.interpolation as interpolation
import eko.alpha_s as alpha_s
from eko.kernel_generation import KernelDispatcher
from eko.constants import Constants


def get_plot(
    lnx,
    ker,
    reDelta=0.03,
    imDelta=0.06,
    reMin=0.01,
    reMax=3.0,
    imMin=0.1,
    imMax=6.0,
    title=None,
    plot_ReIm=True,
):
    """Plot kernel at inversion point

    Parameters
    ---------
        lnx : float
            logarithm of inversion point
        ker : callable
            plotted kernel

    Returns
    -------
        fig : plt.Figure
            generated figure
    """
    # generate 2 2d grids for the x & y bounds
    res, ims = np.mgrid[
        slice(reMin, reMax + reDelta, reDelta), slice(imMin, imMax + imDelta, imDelta)
    ]
    Ns = res + ims * 1j
    vals = []
    for l in Ns:
        r = []
        for N in l:
            r.append(ker(N, lnx))
        vals.append(r)
    vals = np.array(vals)
    # plot
    # # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    vals = vals[:-1, :-1]
    fig = plt.figure(figsize=(10, 10 if plot_ReIm else 5))
    if title is not None:
        t = fig.suptitle(title)
        t.set_in_layout(False)

    n_rows = 2 if plot_ReIm else 1
    if plot_ReIm:
        ax1 = plt.subplot(n_rows, 2, 1)
        im0 = ax1.pcolormesh(res, ims, np.real(vals))
        fig.colorbar(im0, ax=ax1)
        ax1.set_title("Re")

        ax2 = plt.subplot(n_rows, 2, 2)
        im1 = ax2.pcolormesh(res, ims, np.imag(vals))
        fig.colorbar(im1, ax=ax2)
        ax2.set_title("Im")

    ax3 = plt.subplot(n_rows, 2, n_rows * 2 - 1)
    im0 = ax3.pcolormesh(res, ims, np.log(np.abs(vals)))
    fig.colorbar(im0, ax=ax3)
    ax3.set_title("Log(Abs)")

    ax4 = plt.subplot(n_rows, 2, 2 * n_rows)
    im0 = ax4.pcolormesh(res, ims, np.angle(vals), cmap=plt.get_cmap("hsv"))
    fig.colorbar(im0, ax=ax4)
    ax4.set_title("Angle")

    fig.tight_layout()

    return fig


class PathOpt:
    """Helper class"""

    def __init__(self, setup):
        """Constructor

        Parameters
        -----------
            setup : dict
                DGLAP setup dictionary
        """
        self.setup = setup
        # run
        nf = setup["NfFF"]
        mu2init = pow(setup["Q0"], 2)
        mu2final = setup["Q2grid"][0]
        constants = Constants()

        self.basis_function_dispatcher = interpolation.InterpolatorDispatcher(
            setup["xgrid"], setup["xgrid_polynom_rank"], log=True
        )
        delta_t = alpha_s.get_evolution_params(setup, constants, nf, mu2init, mu2final)
        kernel_dispatcher = KernelDispatcher(
            self.basis_function_dispatcher, constants, nf, delta_t, False
        )
        # Receive all precompiled kernels
        self.kernels_ns = kernel_dispatcher.compile_nonsinglet()
        self.kernels_s = kernel_dispatcher.compile_singlet()

    def _save_plots_var(self, ks, kers, xInvs, path):
        """Plot N-space for given basis functions at given inversion points.

        Parameters
        -----------
            ks : array
                list of basis function numbers
            kers : array
                list of kernels
            xInvs : array
                list of inversion points
            path : string
                path prefix
        """
        # iterate basis functions
        for k in ks:
            xk = self.setup["xgrid"][k]
            bf = self.basis_function_dispatcher[k]
            areas = bf.areas_to_const()
            xmin = np.exp(areas[0][0])
            xmax = np.exp(areas[-1][1])
            # iterate inversion points
            for j, xInv in enumerate(xInvs):
                print(f"k={k}, j={j}")
                title = f"k={k}->[{xmin:.2e}<-{xk:.2e}->{xmax:.2e}], xInf={xInv:.2e}"
                fig = get_plot(
                    np.log(xInv),
                    kers[k],
                    reMin=-2,
                    reMax=4,
                    reDelta=0.06,
                    title=title,
                    plot_ReIm=False,
                )
                fig.savefig(path + f"plot-{k}-{j}.png")
                plt.close(fig)

    def save_plots(self, ks, xInvs, var, path):
        """Plots a kernel in N-space for a given set of basis functions
        at a given set of inversion points.

        Parameters
        -----------
            ks : array
                list of basis function numbers
            xInvs : array
                list of inversion points
            var : string
                plotted kernel
            path : string
                path prefix
        """
        kers = None
        singlet_keys = ["S.S", "S.g", "g.S", "g.g"]
        if var == "V.V":
            kers = self.kernels_ns
        elif var in singlet_keys:
            indx = singlet_keys.index(var)
            kers = [kers[indx] for kers in self.kernels_s]
        else:
            raise ValueError("Unkown variable name!")
        self._save_plots_var(ks, kers, xInvs, path)

    def join_plots(self, ks, xInvs, path, totName):
        """Join all plots.

        Parameters
        ---------
            ks : array
                list of basis function numbers
            xInvs : array
                list of inversion points
            path : string
                path prefix
            totName : string
                output file name
        """
        # determine size
        i0 = Image.open(path + f"plot-{ks[0]}-0.png")
        w0, h0 = i0.width, i0.height
        # recombine
        pad = 5
        w, h = w0 + pad, h0 + pad
        dst = Image.new("RGB", (w * len(ks), h * len(xInvs)))
        for nk, k in enumerate(ks):
            for j in range(len(xInvs)):
                i1 = Image.open(path + f"plot-{k}-{j}.png")
                dst.paste(i1, (w * nk, h * j))
        dst.save(path + totName)


if __name__ == "__main__":
    # setup
    n_low = 10
    n_mid = 5
    polynom_rank = 4
    run_init = False
    run_FFNS = False
    run_ZMVFNS = True
    plot_PDF = True
    plot_operator = True

    # combine grid
    flag = f"l{n_low}m{n_mid}r{polynom_rank}"
    xgrid_low = interpolation.get_xgrid_linear_at_log(n_low, 1e-7, 0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n_mid, 0.1, 1.0)
    xgrid_high = np.array([])
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))
    print("xgrid = ", xgrid, " [", len(xgrid), "]")
    targetgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko.dglap").handlers = []
    logging.getLogger("eko.dglap").addHandler(logStdout)
    logging.getLogger("eko.dglap").setLevel(logging.DEBUG)

    setup = {
        "PTO": 0,
        "alphas": 0.35,
        "Qref": np.sqrt(2),
        "Q0": np.sqrt(2),
        "FNS": "FFNS",
        "NfFF": 4,
        "xgrid_type": "custom",
        "xgrid": xgrid,
        "xgrid_polynom_rank": polynom_rank,
        "targetgrid": targetgrid,
        "Q2grid": [1e4],
    }

    app = PathOpt(setup)

    run_imgs = False
    run_join_imgs = True
    ks = [2, 4, 6, 8, 10, 12]
    xInvs = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    # iterate all operators
    for op_name, path in [
        ("V.V", "NS/"),
        ("S.S", "S_qq/"),
        ("S.g", "S_qg/"),
        ("g.S", "S_gq/"),
        ("g.g", "S_gg/"),
    ]:
        print(f"write {op_name} to '{path}'")
        # build all imgs
        if run_imgs:
            app.save_plots(ks, xInvs, op_name, path)
        # join all imgs
        if run_join_imgs:
            app.join_plots(ks, xInvs, path, path[:-1] + ".png")
