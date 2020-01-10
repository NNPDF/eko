# -*- coding: utf-8 -*-
import logging
import sys
from collections import abc
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PIL import Image

import eko.interpolation as interpolation
import eko.alpha_s as alpha_s
from eko.kernel_generation import KernelDispatcher
from eko.constants import Constants


def get_area_plot(
    lnx,
    ker,
    reDelta=0.03,
    imDelta=0.06,
    reMin=0.01,
    reMax=3.0,
    imMin=0.1,
    imMax=6.0,
    title=None,
    plot_ReIm=False,
):
    """Plot kernel in N-space at given inversion point

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


def get_re_plot(
    lnx,
    ker,
    reN=100,
    reMin=1.1,
    reMax=10.0,
    title=None
):
    """Plot kernel on real axis at given inversion point

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
    # generate data
    res = np.logspace(np.log10(reMin),np.log10(reMax),num=reN) #np.arange(reMin, reMax+reDelta, reDelta)
    vals = []
    for r in res:
        vals.append(ker(r, lnx))
    vals = np.real(np.array(vals))
    # plot
    fig = plt.figure(figsize=(5,5))
    if title is not None:
        t = fig.suptitle(title)
        t.set_in_layout(False)
    plt.loglog(res,vals)
    plt.loglog(res,0.0-vals)
    #fig.tight_layout()
    return fig


class PathOpt:
    """Helper class"""

    def __init__(self, setup, ks, xInvs):
        """Constructor

        Parameters
        -----------
            setup : dict
                DGLAP setup dictionary
            ks : array
                list of basis function numbers
            xInvs : array
                list of inversion points
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
            self.basis_function_dispatcher, constants, nf, delta_t, numba_it=False
        )
        # Receive all precompiled kernels
        self.kernels_ns = kernel_dispatcher.compile_nonsinglet()
        self.kernels_s = kernel_dispatcher.compile_singlet()
        # set params
        self.ks = ks
        self.xInvs = xInvs

    def _get_kers(self, op_name):
        """Collect all kernels for operator

        Parameters
        -----------
            op_name : string
                plotted operator

        Returns
        --------
            kers : array
                list of kernels
        """
        kers = None
        singlet_keys = ["S.S", "S.g", "g.S", "g.g"]
        if op_name == "V.V":
            kers = self.kernels_ns
        elif op_name in singlet_keys:
            indx = singlet_keys.index(op_name)
            kers = [kers[indx] for kers in self.kernels_s]
        else:
            raise ValueError(f"Unkown operator '{op_name}'!")
        return kers

    def save_plots(self, op_name, path, plot_type):
        """Plots all kernels.

        Parameters
        -----------
            op_name : string
                plotted operator
            path : string
                path prefix
            plot_type : {'area', 're'}
                diagram type
        """
        # collect kernels
        kers = self._get_kers(op_name)

        print(f"plotting {plot_type} ...")
        # iterate basis functions
        for k in self.ks:
            xk = self.setup["xgrid"][k]
            bf = self.basis_function_dispatcher[k]
            areas = bf.areas_to_const()
            xmin = np.exp(areas[0][0])
            xmax = np.exp(areas[-1][1])
            # iterate inversion points
            for j, xInv in enumerate(self.xInvs):
                print(f"k={k}, j={j}")
                title = f"k={k}->[{xmin:.2e}<-{xk:.2e}->{xmax:.2e}], xInv={xInv:.2e}"
                fig = None
                out_name = None
                # run
                if plot_type == "area":
                    fig = get_area_plot(
                        np.log(xInv),
                        kers[k],
                        reMin=-2,
                        reMax=14,
                        reDelta=0.16,
                        imMax=12,
                        imDelta=0.12,
                        title=title,
                        plot_ReIm=False,
                    )
                    out_name = f"area-{k}-{j}.png"
                elif plot_type == "re":
                    fig = get_re_plot(
                        np.log(xInv),
                        kers[k],
                        title=title,
                        reMin=0.2 if op_name == "V.V" else 1.1,
                        reMax=40,
                        reN=100
                    )
                    out_name = f"re-{k}-{j}.png"
                else:
                    raise ValueError(f"Unknown plot type '{plot_type}'!")
                # write
                fig.savefig(path + out_name)
                plt.close(fig)

    def join_plots(self, path, totName, plot_type):
        """Joins all plots.

        Parameters
        ---------
            path : string
                path prefix
            totName : string
                output file name
            plot_type : {'area', 're'}
                diagram type
        """
        fn = None
        if plot_type == "area":
            fn = "area-{k}-{j}.png"
        elif plot_type == "re":
            fn = "re-{k}-{j}.png"
        else:
            raise ValueError(f"Unknown plot type '{plot_type}'!")
        # determine size
        k0 = self.ks[0]
        i0 = Image.open(path + fn.format(k=k0,j=0))
        w0, h0 = i0.width, i0.height
        # recombine
        pad = 5
        w, h = w0 + pad, h0 + pad
        dst = Image.new("RGB", (w * len(ks), h * len(self.xInvs)))
        for nk, k in enumerate(self.ks):
            for j in range(len(self.xInvs)):
                i1 = Image.open(path + fn.format(k=k,j=j))
                dst.paste(i1, (w * nk, h * j))
        dst.save(path + totName)

    def plot_mins(self, op_name, out_name):
        """Plots all minima

        Parameters
        -----------
            op_name : string
                plotted operator
            out_name : string
                output file name
        """
        # collect kernels
        kers = self._get_kers(op_name)
        print(f"searching minima of {op_name}")
        fig = plt.figure(figsize=(7,7))
        plt.suptitle(f"minima of {op_name} along real axis")
        # iterate basis functions
        for k in self.ks:
            bf = self.basis_function_dispatcher[k]
            areas = bf.areas_to_const()
            lnxmax = areas[-1][1]
            xmax = np.exp(lnxmax)
            x_mins = []
            # iterate inversion points
            for j, xInv in enumerate(self.xInvs):
                lnxInv = np.log(xInv)
                # skip?
                if xInv >= xmax or lnxInv >= lnxmax:
                    continue
                # find minimum
                def f(r,lnx):
                    if isinstance(r,abc.Iterable):
                        return np.array([f(e,lnx) for e in r])
                    return np.real(kers[k](r,lnx))
                r_min = 0 if op_name == "V.V" else 1
                r_min += 0.1
                r_max = 6.+2.6*(10.+lnxInv)
                mi = minimize(f,2,args=(lnxInv,),bounds=[(r_min,r_max)])
                if not mi.success:
                    print(mi)
                    x_mins.append(np.NaN)
                else:
                    x_mins.append(mi.x[0])
            plt.semilogx(self.xInvs[:len(x_mins)],np.array(x_mins),marker="o")
            print(x_mins)
        def guess(x):
            return 1.3 + np.power(x,1.5)*25
        plt.semilogx(self.xInvs,guess(self.xInvs),color="black")
        # write
        plt.xlabel("Inversion point")
        fig.savefig(out_name)
        plt.close(fig)

if __name__ == "__main__":
    # setup
    n_low = 10
    n_mid = 5
    polynom_rank = 4
    run_area_imgs = False
    run_re_imgs = True
    run_join_area_imgs = False
    run_join_re_imgs = True
    run_mins = False
    ks = [2, 4, 6, 8, 10, 12]
    xInvs = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    # combine grid
    #flag = f"l{n_low}m{n_mid}r{polynom_rank}"
    xgrid_low = interpolation.get_xgrid_linear_at_log(n_low, 1e-7, 0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n_mid, 0.1, 1.0)
    xgrid_high = np.array([])
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))
    print("xgrid = ", xgrid, " [", len(xgrid), "]")
    targetgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])
    ks = list(range(len(xgrid)))
    xInvs = targetgrid

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

    app = PathOpt(setup, ks, xInvs)

    # iterate all operators
    for op_name, path in [
        ("V.V", "NS/"),
        ("S.S", "S_qq/"),
        ("S.g", "S_qg/"),
        ("g.S", "S_gq/"),
        ("g.g", "S_gg/"),
    ][:1]:
        print(f"run {op_name} with '{path}'")
        # build all imgs
        if run_area_imgs:
            app.save_plots(op_name, path, "area")
        if run_re_imgs:
            app.save_plots(op_name, path, "re")
        # join all imgs
        if run_join_area_imgs:
            app.join_plots(path, "area-"+path[:-1] + ".png", "area")
        if run_join_re_imgs:
            app.join_plots(path, "re-"+path[:-1] + ".png", "re")
        # minima
        if run_mins:
            app.plot_mins(op_name, "mins-"+path[:-1]+".png")
