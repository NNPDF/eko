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

def get_plot(k, lnx, ker,reDelta=0.03,imDelta=0.06,reMin=0.01,reMax=3.,imMin=0.1,imMax=6.,title=None):
    # generate 2 2d grids for the x & y bounds
    res, ims = np.mgrid[slice(reMin, reMax + reDelta, reDelta),slice(imMin, imMax + imDelta, imDelta)]
    Ns = res + ims*1j
    vals = []
    for l in Ns:
        r = []
        for N in l:
            r.append(ker(N,lnx))
        vals.append(r)
    vals = np.array(vals)
    # plot
    # # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    vals = vals[:-1, :-1]
    fig = plt.figure(figsize=(10, 10))
    if title is not None:
        t = fig.suptitle(title)
        t.set_in_layout(False)

    ax1 = plt.subplot(2,2,1)
    im0 = ax1.pcolormesh(res,ims,np.real(vals))
    fig.colorbar(im0, ax=ax1)
    ax1.set_title('Re')

    ax2 = plt.subplot(2,2,2)
    im1 = ax2.pcolormesh(res,ims,np.imag(vals))
    fig.colorbar(im1, ax=ax2)
    ax2.set_title('Im')

    ax3 = plt.subplot(2,2,3)
    im0 = ax3.pcolormesh(res,ims,np.log(np.abs(vals)))
    fig.colorbar(im0, ax=ax3)
    ax3.set_title('Log(Abs)')

    ax4 = plt.subplot(2,2,4)
    im0 = ax4.pcolormesh(res,ims,np.angle(vals),cmap=plt.get_cmap("hsv"))
    fig.colorbar(im0, ax=ax4)
    ax4.set_title('Angle')
    
    fig.tight_layout()

    return fig


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
    print("xgrid = ",xgrid," [",len(xgrid),"]")
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

    # run
    nf = setup["NfFF"]
    mu2init = pow(setup["Q0"],2)
    mu2final = setup["Q2grid"][0]
    constants = Constants()

    basis_function_dispatcher = interpolation.InterpolatorDispatcher(
        xgrid, polynom_rank, log=True
    )
    delta_t = alpha_s.get_evolution_params(setup, constants, nf, mu2init, mu2final)
    kernel_dispatcher = KernelDispatcher(
        basis_function_dispatcher, constants, nf, delta_t,False
    )
     # Receive all precompiled kernels
    kernels = kernel_dispatcher.compile_nonsinglet()

    run_imgs = True
    run_join_imgs = True

    ks = [2,4,6,8,10,12]
    xInvs = [1e-4,1e-3,1e-2,.1,.2,.4,.6,.8,.9]
    # build all imgs
    if run_imgs:
        for k in ks:
            bf = basis_function_dispatcher[k]
            areas = bf.areas_to_const()
            xmin = np.exp(areas[0][0])
            xmax = np.exp(areas[-1][1])
            reMax = 4.
            reDelta = .04
            for j,xInv in enumerate(xInvs):
                print(f"k={k}, j={j}")
                title = f"k={k}->[{xmin:.2e},{xmax:.2e}], xInf={xInv:.2e}"
                fig = get_plot(k,np.log(xInv),kernels[k],reMax=reMax,reDelta=reDelta,title=title)
                fig.savefig(f"plot-{k}-{j}.png")
                plt.close(fig)
    # join all imgs
    if run_join_imgs:
        dst = Image.new('RGB',(1005*len(ks),1005*len(xInvs)))
        for nk,k in enumerate(ks):
            for j,xInv in enumerate(xInvs):
                i1 = Image.open(f"plot-{k}-{j}.png")
                dst.paste(i1,(1005*nk,1005*j))
        dst.save("tot.png")