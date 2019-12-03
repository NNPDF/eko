# -*- coding: utf-8 -*-
import logging
import sys
import pathlib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import eko.dglap as dglap
import eko.interpolation as interpolation
from tools import plot_dist, save_all_operators_to_pdf


toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])
# implement Eq. 31 of arXiv:hep-ph/0204316
def toy_uv0(x):
    return 5.107200 * x ** (0.8) * (1.0 - x) ** 3 / x


def toy_dv0(x):
    return 3.064320 * x ** (0.8) * (1.0 - x) ** 4 / x


def toy_g0(x):
    return 1.7 * x ** (-0.1) * (1.0 - x) ** 5 / x


def toy_dbar0(x):
    return 0.1939875 * x ** (-0.1) * (1.0 - x) ** 6 / x


def toy_ubar0(x):
    return (1.0 - x) * toy_dbar0(x)


def toy_s0(x):
    return 0.2 * (toy_ubar0(x) + toy_dbar0(x))


def toy_sbar0(x):
    return toy_s0(x)


def toy_Lm0(x):
    return toy_dbar0(x) - toy_ubar0(x)


def toy_Lp0(x):
    return (toy_dbar0(x) + toy_ubar0(x)) * 2.0  # 2 is missing in the paper!


def toy_sp0(x):
    return toy_s0(x) + toy_sbar0(x)


def toy_T30(x):
    return -2.0 * toy_Lm0(x) + toy_uv0(x) - toy_dv0(x)


def toy_T80(x):
    return toy_Lp0(x) + toy_uv0(x) + toy_dv0(x) - 2.0 * toy_sp0(x)


def toy_S0(x):
    return toy_uv0(x) + toy_dv0(x) + toy_Lp0(x) + toy_sp0(x)


def save_initial_scale_plots_to_pdf(path):
    pp = PdfPages(path)
    # check table 2 part 1 of arXiv:hep-ph/0204316
    toy_uv0_grid = np.array([toy_uv0(x) for x in toy_xgrid])
    toy_xuv0_grid_ref = np.array(
        [
            1.2829e-5,
            8.0943e-5,
            5.1070e-4,
            3.2215e-3,
            2.0271e-2,
            1.2448e-1,
            5.9008e-1,
            6.6861e-1,
            3.6666e-1,
            1.0366e-1,
            4.6944e-3,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_uv0_grid,
        toy_xuv0_grid_ref,
        title="xu_v(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    toy_dv0_grid = np.array([toy_dv0(x) for x in toy_xgrid])
    toy_xdv0_grid_ref = np.array(
        [
            7.6972e-6,
            4.8566e-5,
            3.0642e-4,
            1.9327e-3,
            1.2151e-2,
            7.3939e-2,
            3.1864e-1,
            2.8082e-1,
            1.1000e-1,
            1.8659e-2,
            2.8166e-4,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_dv0_grid,
        toy_xdv0_grid_ref,
        title="xd_v(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    toy_Lm0_grid = np.array([toy_Lm0(x) for x in toy_xgrid])
    toy_xLm0_grid_ref = np.array(
        [
            9.7224e-8,
            7.7227e-7,
            6.1341e-6,
            4.8698e-5,
            3.8474e-4,
            2.8946e-3,
            1.2979e-2,
            7.7227e-3,
            1.6243e-3,
            1.0259e-4,
            1.7644e-7,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_Lm0_grid,
        toy_xLm0_grid_ref,
        title="xL_-(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    toy_Lp0_grid = np.array([toy_Lp0(x) for x in toy_xgrid])
    toy_xLp0_grid_ref = np.array(
        [
            3.8890e0,
            3.0891e0,
            2.4536e0,
            1.9478e0,
            1.5382e0,
            1.1520e0,
            4.9319e-1,
            8.7524e-2,
            9.7458e-3,
            3.8103e-4,
            4.3129e-7,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_Lp0_grid,
        toy_xLp0_grid_ref,
        title="xL_+(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    toy_sp0_grid = np.array([toy_sp0(x) for x in toy_xgrid])
    toy_xsp0_grid_ref = np.array(
        [
            7.7779e-1,
            6.1782e-1,
            4.9072e-1,
            3.8957e-1,
            3.0764e-1,
            2.3041e-1,
            9.8638e-2,
            1.7505e-2,
            1.9492e-3,
            7.6207e-5,
            8.6259e-8,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_sp0_grid,
        toy_xsp0_grid_ref,
        title="xs_+(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    toy_g0_grid = np.array([toy_g0(x) for x in toy_xgrid])
    toy_xg0_grid_ref = np.array(
        [
            8.5202e0,
            6.7678e0,
            5.3756e0,
            4.2681e0,
            3.3750e0,
            2.5623e0,
            1.2638e0,
            3.2228e-1,
            5.6938e-2,
            4.2810e-3,
            1.7180e-5,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_g0_grid,
        toy_xg0_grid_ref,
        title="xg(x,µ_F^2 = 2 GeV^2)",
    )
    pp.savefig()

    # close
    pp.close()


# check table 2 part 2 of arXiv:hep-ph/0204316
def save_table2_2_to_pdf(path, ret1):
    pp = PdfPages(path)

    # u_v
    toy_uv1_xgrid = np.array([toy_uv0(x) for x in ret1["xgrid"]])
    toy_uv1_grid = np.dot(ret1["operators"]["NS"], toy_uv1_xgrid)
    toy_xuv1_grid_ref = np.array(
        [
            5.7722e-5,
            3.3373e-4,
            1.8724e-3,
            1.0057e-2,
            5.0392e-2,
            2.1955e-1,
            5.7267e-1,
            3.7925e-1,
            1.3476e-1,
            2.3123e-2,
            4.3443e-4,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_uv1_grid,
        toy_xuv1_grid_ref,
        title="xu_v(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()

    # d_v
    toy_dv1_xgrid = np.array([toy_dv0(x) for x in ret1["xgrid"]])
    toy_dv1_grid = np.dot(ret1["operators"]["NS"], toy_dv1_xgrid)
    toy_xdv1_grid_ref = np.array(
        [
            3.4343e-5,
            1.9800e-4,
            1.1065e-3,
            5.9076e-3,
            2.9296e-2,
            1.2433e-1,
            2.8413e-1,
            1.4186e-1,
            3.5364e-2,
            3.5943e-3,
            2.2287e-5,
        ]
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_dv1_grid,
        toy_xdv1_grid_ref,
        title="xd_v(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()

    # T3, i.e. L-
    toy_T31_xgrid = np.array([toy_T30(x) for x in ret1["xgrid"]])
    toy_T31_grid = np.dot(ret1["operators"]["NS"], toy_T31_xgrid)
    toy_xLm1_grid_ref = np.array(
        [
            7.6527e-7,
            5.0137e-6,
            3.1696e-5,
            1.9071e-4,
            1.0618e-3,
            4.9731e-3,
            1.0470e-2,
            3.3029e-3,
            4.2815e-4,
            1.5868e-5,
            1.1042e-8,
        ]
    )
    toy_xT31_grid_ref = -2.0 * toy_xLm1_grid_ref + toy_xuv1_grid_ref - toy_xdv1_grid_ref
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_T31_grid,
        toy_xT31_grid_ref,
        title="xT_3(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()

    # T8, i.e. s+
    toy_T81_xgrid = np.array([toy_T80(x) for x in ret1["xgrid"]])
    toy_T81_grid = np.dot(ret1["operators"]["NS"], toy_T81_xgrid)
    toy_xLp1_grid_ref = np.array(
        [
            9.9465e1,
            5.0259e1,
            2.4378e1,
            1.1323e1,
            5.0324e0,
            2.0433e0,
            4.0832e-1,
            4.0165e-2,
            2.8624e-3,
            6.8961e-5,
            3.6293e-8,
        ]
    )
    toy_xsp1_grid_ref = np.array(
        [
            4.8642e1,
            2.4263e1,
            1.1501e1,
            5.1164e0,
            2.0918e0,
            7.2814e-1,
            1.1698e-1,
            1.0516e-2,
            7.3138e-4,
            1.7725e-5,
            1.0192e-8,
        ]
    )
    toy_xT81_grid_ref = (
        toy_xLp1_grid_ref
        + toy_xuv1_grid_ref
        + toy_xdv1_grid_ref
        - 2.0 * toy_xsp1_grid_ref
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_T81_grid,
        toy_xT81_grid_ref,
        title="xT_8(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()

    # T15, i.e. c+
    toy_T151_xgrid = np.array([toy_S0(x) for x in ret1["xgrid"]])
    toy_T151_grid = np.dot(ret1["operators"]["NS"], toy_T151_xgrid)
    toy_xcp1_grid_ref = np.array(
        [
            4.7914e1,
            2.3685e1,
            1.1042e1,
            4.7530e0,
            1.8089e0,
            5.3247e-1,
            5.8864e-2,
            4.1379e-3,
            2.6481e-4,
            6.5549e-6,
            4.8893e-9,
        ]
    )
    toy_T151_grid_ref = (
        toy_xLp1_grid_ref
        + toy_xuv1_grid_ref
        + toy_xdv1_grid_ref
        + toy_xsp1_grid_ref
        - 3.0 * toy_xcp1_grid_ref
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_T151_grid,
        toy_T151_grid_ref,
        title="xT_15(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()

    # Singlet + gluon
    toy_S1_xgrid = np.array([toy_S0(x) for x in ret1["xgrid"]])
    toy_g1_xgrid = np.array([toy_g0(x) for x in ret1["xgrid"]])
    toy_S1_grid = np.dot(ret1["operators"]["S_qq"], toy_S1_xgrid) + np.dot(
        ret1["operators"]["S_qg"], toy_g1_xgrid
    )
    toy_g1_grid = np.dot(ret1["operators"]["S_gq"], toy_S1_xgrid) + np.dot(
        ret1["operators"]["S_gg"], toy_g1_xgrid
    )
    toy_xg1_grid_ref = np.array(
        [
            1.3162e3,
            6.0008e2,
            2.5419e2,
            9.7371e1,
            3.2078e1,
            8.0546e0,
            8.8766e-1,
            8.2676e-2,
            7.9240e-3,
            3.7311e-4,
            1.0918e-6,
        ]
    )
    toy_xS1_grid_ref = (
        toy_xuv1_grid_ref
        + toy_xdv1_grid_ref
        + toy_xLp1_grid_ref
        + toy_xsp1_grid_ref
        + toy_xcp1_grid_ref
    )
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_S1_grid,
        toy_xS1_grid_ref,
        title="xSigma(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()
    plot_dist(
        toy_xgrid,
        toy_xgrid * toy_g1_grid,
        toy_xg1_grid_ref,
        title="xg(x,µ_F^2 = 10^4 GeV^2)",
    )
    pp.savefig()
    # close
    pp.close()


# output path
assets_path = pathlib.Path(__file__).with_name("assets")

if __name__ == "__main__":
    # setup
    n_low = 10
    n_mid = 5
    polynom_rank = 4
    flag = f"l{n_low}m{n_mid}r{polynom_rank}"
    xgrid_low = interpolation.get_xgrid_linear_at_log(n_low, 1e-7, 0.1)
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n_mid, 0.1, 1.0)
    xgrid_high = np.array([])
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid, xgrid_high)))

    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko.dglap").handlers = []
    logging.getLogger("eko.dglap").addHandler(logStdout)
    logging.getLogger("eko.dglap").setLevel(logging.DEBUG)

    # run
    ret = dglap.run_dglap(
        {
            "PTO": 0,
            "alphas": 0.35,
            "Qref": np.sqrt(2),
            "Q0": np.sqrt(2),
            "NfFF": 4,
            "xgrid_type": "custom",
            "xgrid": xgrid,
            "xgrid_polynom_rank": polynom_rank,
            "targetgrid": toy_xgrid,
            "Q2grid": [1e4],
        }
    )

    # compare and save
    print("compare and save to file ...")
    # save_initial_scale_plots_to_pdf(assets_path / f"LHA-LO-FFNS-init-{flag}.pdf")
    save_table2_2_to_pdf(assets_path / f"LHA-LO-FFNS-plots-{flag}.pdf", ret)
    save_all_operators_to_pdf(assets_path / f"LHA-LO-FFNS-ops-{flag}.pdf", ret)
