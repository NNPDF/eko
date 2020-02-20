# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import copy
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import eko.dglap as dglap
import eko.interpolation as interpolation
import eko.utils as utils

from tools import plot_dist, save_all_operators_to_pdf

# xgrid
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


def toy_cp0(x):  # pylint: disable=unused-argument
    return 0


def toy_bp0(x):  # pylint: disable=unused-argument
    return 0


def toy_V0(x):
    return toy_uv0(x) + toy_dv0(x)


def toy_V30(x):
    return toy_uv0(x) - toy_dv0(x)


def toy_T30(x):
    return -2.0 * toy_Lm0(x) + toy_uv0(x) - toy_dv0(x)


def toy_T80(x):
    return toy_Lp0(x) + toy_uv0(x) + toy_dv0(x) - 2.0 * toy_sp0(x)


def toy_S0(x):
    return toy_uv0(x) + toy_dv0(x) + toy_Lp0(x) + toy_sp0(x)


# collect pdfs
LHA_init_pdfs = {
    "V": toy_V0,
    "V3": toy_V30,
    "T3": toy_T30,
    "T8": toy_T80,
    "T15": toy_S0,
    "S": toy_S0,
    "g": toy_g0,
}

# list
raw_label_list = ["u_v", "d_v", "L_-", "L_+", "s_+", "c_+", "b_+", "g"]
rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

# fmt: off
# inital reference grid = table 2 part 1
void = np.zeros(len(toy_xgrid))
LHA_init_grid_ref = np.array([
    [1.2829e-5,8.0943e-5,5.1070e-4,3.2215e-3,2.0271e-2,1.2448e-1,5.9008e-1,6.6861e-1,3.6666e-1,1.0366e-1,4.6944e-3], # u_v # pylint: disable=line-too-long
    [7.6972e-6,4.8566e-5,3.0642e-4,1.9327e-3,1.2151e-2,7.3939e-2,3.1864e-1,2.8082e-1,1.1000e-1,1.8659e-2,2.8166e-4], # d_v # pylint: disable=line-too-long
    [9.7224e-8,7.7227e-7,6.1341e-6,4.8698e-5,3.8474e-4,2.8946e-3,1.2979e-2,7.7227e-3,1.6243e-3,1.0259e-4,1.7644e-7], # L_- # pylint: disable=line-too-long
    [3.8890e+0,3.0891e+0,2.4536e+0,1.9478e+0,1.5382e+0,1.1520e+0,4.9319e-1,8.7524e-2,9.7458e-3,3.8103e-4,4.3129e-7], # L_+ # pylint: disable=line-too-long
    [7.7779e-1,6.1782e-1,4.9072e-1,3.8957e-1,3.0764e-1,2.3041e-1,9.8638e-2,1.7505e-2,1.9492e-3,7.6207e-5,8.6259e-8], # s_+ # pylint: disable=line-too-long
    void, # c_+
    void, # b_+
    [8.5202e+0,6.7678e+0,5.3756e+0,4.2681e+0,3.3750e+0,2.5623e+0,1.2638e+0,3.2228e-1,5.6938e-2,4.2810e-3,1.7180e-5], # g # pylint: disable=line-too-long
])
# my/exact initial grid
LHA_init_grid = []
for f in [toy_uv0, toy_dv0, toy_Lm0, toy_Lp0, toy_sp0, toy_cp0, toy_bp0, toy_g0]:
    LHA_init_grid.append(f(toy_xgrid))
LHA_init_grid = np.array(LHA_init_grid)

# reference grid at final for FFNS = table 2 part 2
LHA_final_grid_FFNS_ref = np.array([
    [5.7722e-5,3.3373e-4,1.8724e-3,1.0057e-2,5.0392e-2,2.1955e-1,5.7267e-1,3.7925e-1,1.3476e-1,2.3123e-2,4.3443e-4], # u_v # pylint: disable=line-too-long
    [3.4343e-5,1.9800e-4,1.1065e-3,5.9076e-3,2.9296e-2,1.2433e-1,2.8413e-1,1.4186e-1,3.5364e-2,3.5943e-3,2.2287e-5], # d_v # pylint: disable=line-too-long
    [7.6527e-7,5.0137e-6,3.1696e-5,1.9071e-4,1.0618e-3,4.9731e-3,1.0470e-2,3.3029e-3,4.2815e-4,1.5868e-5,1.1042e-8], # L_- # pylint: disable=line-too-long
    [9.9465e+1,5.0259e+1,2.4378e+1,1.1323e+1,5.0324e+0,2.0433e+0,4.0832e-1,4.0165e-2,2.8624e-3,6.8961e-5,3.6293e-8], # L_+ # pylint: disable=line-too-long
    [4.8642e+1,2.4263e+1,1.1501e+1,5.1164e+0,2.0918e+0,7.2814e-1,1.1698e-1,1.0516e-2,7.3138e-4,1.7725e-5,1.0192e-8], # s_+ # pylint: disable=line-too-long
    [4.7914e+1,2.3685e+1,1.1042e+1,4.7530e+0,1.8089e+0,5.3247e-1,5.8864e-2,4.1379e-3,2.6481e-4,6.5549e-6,4.8893e-9], # c_+ # pylint: disable=line-too-long
    void, # b_+
    [1.3162e+3,6.0008e+2,2.5419e+2,9.7371e+1,3.2078e+1,8.0546e+0,8.8766e-1,8.2676e-2,7.9240e-3,3.7311e-4,1.0918e-6], # g # pylint: disable=line-too-long
])

# reference grid at final for ZM-VFNS = table 2 part 3
LHA_final_grid_ZMVFNS_ref = np.array([
    [5.8771e-5,3.3933e-4,1.9006e-3,1.0186e-2,5.0893e-2,2.2080e-1,5.7166e-1,3.7597e-1,1.3284e-1,2.2643e-2,4.2047e-4], # u_v # pylint: disable=line-too-long
    [3.4963e-5,2.0129e-4,1.1229e-3,5.9819e-3,2.9576e-2,1.2497e-1,2.8334e-1,1.4044e-1,3.4802e-2,3.5134e-3,2.1529e-5], # d_v # pylint: disable=line-too-long
    [7.8233e-7,5.1142e-6,3.2249e-5,1.9345e-4,1.0730e-3,4.9985e-3,1.0428e-2,3.2629e-3,4.2031e-4,1.5468e-5,1.0635e-8], # L_- # pylint: disable=line-too-long
    [1.0181e+2,5.1182e+1,2.4693e+1,1.1406e+1,5.0424e+0,2.0381e+0,4.0496e-1,3.9592e-2,2.8066e-3,6.7201e-5,3.4998e-8], # L_+ # pylint: disable=line-too-long
    [4.9815e+1,2.4725e+1,1.1659e+1,5.1583e+0,2.0973e+0,7.2625e-1,1.1596e-1,1.0363e-2,7.1707e-4,1.7278e-5,9.8394e-9], # s_+ # pylint: disable=line-too-long
    [4.9088e+1,2.4148e+1,1.1201e+1,4.7953e+0,1.8147e+0,5.3107e-1,5.8288e-2,4.0740e-3,2.5958e-4,6.3958e-6,4.7330e-9], # c_+ # pylint: disable=line-too-long
    [4.6070e+1,2.2239e+1,1.0037e+1,4.1222e+0,1.4582e+0,3.8106e-1,3.5056e-2,2.2039e-3,1.3522e-4,3.3996e-6,2.8903e-9], # b_+ # pylint: disable=line-too-long
    [1.3272e+3,6.0117e+2,2.5282e+2,9.6048e+1,3.1333e+1,7.7728e+0,8.4358e-1,7.8026e-2,7.4719e-3,3.5241e-4,1.0307e-6], # g # pylint: disable=line-too-long
])

# rotation matrix
LHA_flavour_rotate = np.array([
    # u_v, d_v, L_-, L_+, s_+, c_+, b_+,   g
    [   1,   1,   0,   0,   0,   0,   0,   0], # V
    [   1,  -1,   0,   0,   0,   0,   0,   0], # V3
    [   1,  -1,  -2,   0,   0,   0,   0,   0], # T3
    [   1,   1,   0,   1,  -2,   0,   0,   0], # T8
    [   1,   1,   0,   1,   1,  -3,   0,   0], # T15
    [   1,   1,   0,   1,   1,   1,  -4,   0], # T24
    [   1,   1,   0,   1,   1,   1,   1,   0], # S
    [   0,   0,   0,   0,   0,   0,   0,   1], # g
])
# fmt: on

# rotate basis
def rotate_and_dict(inp):
    rot = np.dot(LHA_flavour_rotate, inp)
    out = {}
    for k, n in enumerate(["V", "V3", "T3", "T8", "T15", "T24", "S", "g"]):
        out[n] = rot[k]
    return out


LHA_final_dict_FFNS_ref = rotate_and_dict(LHA_final_grid_FFNS_ref)
LHA_final_dict_ZMVFNS_ref = rotate_and_dict(LHA_final_grid_ZMVFNS_ref)

# output path
assets_path = pathlib.Path(__file__).with_name("assets")


class LHABenchmarkPaper:
    """
        Compares to the LHA benchmark paper :cite:`Giele:2002hx`.

        Parameters
        ----------
            xgrid : array
                basis grid
            polynomial_degree : int
                degree of interpolation polynomial
            flag : string
                output file tag
    """
    def __init__(self, xgrid, polynomial_degree, flag):
        self._xgrid = xgrid
        self._polynomial_degree = polynomial_degree
        self._flag = flag
        # default config for post processing
        self.post_process_config = {
            "plot_PDF": True,
            "plot_operator": True,
            "write_operator": True,
        }
        # default setup
        self._Q2init = 2
        self._Q2final = 1e4
        self._setup = {
            "PTO": 0,
            "alphas": 0.35,
            "Qref": np.sqrt(2),
            "Qmc": np.sqrt(self._Q2init),
            "Qmb": 4.5,
            "Qmt": 175.0,
            "xgrid_type": "custom",
            "xgrid": xgrid,
            "xgrid_polynom_rank": polynom_rank,
        }

    def _post_process(self, ret, ref, tag):
        """
            Handles the post processing of the run

            Parameters
            ----------
                ret : dict
                    DGLAP result
                tag : string
                    file tag
        """
        if self.post_process_config["plot_PDF"]:
            self._save_final_scale_plots_to_pdf(
                assets_path / f"LHA-LO-{tag}-plots-{flag}.pdf", ret, ref
            )
        if self.post_process_config["plot_operator"]:
            save_all_operators_to_pdf(ret, assets_path / f"LHA-LO-{tag}-ops-{flag}.pdf")
        if self.post_process_config["write_operator"]:
            dglap.write_YAML_to_file(ret, assets_path / f"LHA-LO-{tag}-ops-{flag}.yaml")

    def _run_FFNS_raw(self, tag, Q2init, Q2final):
        """
            Runs a fixed-flavor-number-scheme configuration.

            Parameters
            ----------
                tag : string
                    file tag
                Q2init : t_flaot
                    init scale squared
                Q2final : t_flaot
                    final scale squared

            Returns
            -------
                ret : dict
                    DGLAP result
        """
        add_setup = {
            "FNS": "FFNS",
            "NfFF": 4,
            "Q0": np.sqrt(Q2init),
            "Q2grid": [Q2final],
        }
        # xgrid can be a copy, so we don't need a deep copy here
        setup = utils.merge_dicts(copy.copy(self._setup), add_setup)
        ret = dglap.run_dglap(setup)
        self._post_process(ret, LHA_final_dict_FFNS_ref, tag)
        return ret

    def run_FFNS(self):
        """
            Runs the fixed-flavor-number-scheme configuration.

            Returns
            -------
                ret : dict
                    DGLAP result
        """
        ret = self._run_FFNS_raw("FFNS", self._Q2init, self._Q2final)
        return ret

    def run_FFNS_twostep(self, Q2mid):
        """
            Runs the fixed-flavor-number-scheme configuration with a step in the middle.

            Parameters
            ----------
                Q2mid : t_float
                    break point

            Returns
            -------
                ret2t1 : dict
                    DGLAP result
        """
        # suppress PDF plots for single steps
        plot_PDF = self.post_process_config["plot_PDF"]
        self.post_process_config["plot_PDF"] = False
        # step 1
        step1 = self._run_FFNS_raw("FFNS-twostep-step1", self._Q2init, Q2mid)
        # step 2
        step2 = self._run_FFNS_raw("FFNS-twostep-step2", Q2mid, self._Q2final)
        # join 2*1
        self.post_process_config["plot_PDF"] = plot_PDF
        ret2t1 = dglap.multiply_operators(step2, step1)
        self._post_process(ret2t1, LHA_final_dict_FFNS_ref, "FFNS-twostep-step2t1")
        return ret2t1

    def run_ZMVFNS(self):
        """
            Runs Zero-mass variable-flavor-number-scheme configuration.

            Returns
            -------
                ret : dict
                    DGLAP result
        """
        add_setup = {
            "FNS": "ZM-VFNS",
            "Q0": np.sqrt(self._Q2init),
            "Q2grid": [self._Q2final],
        }
        # xgrid can be copy, so we don't need a deep copy
        setup = utils.merge_dicts(copy.copy(self._setup), add_setup)
        ret = dglap.run_dglap(setup)
        self._post_process(ret, LHA_final_dict_ZMVFNS_ref, "ZMVFNS")
        return ret

    def _save_final_scale_plots_to_pdf(self, path, ret, ref):
        """
            Plots all PDFs at the final scale.

            The reference values are given in Table 2 part 2,3 of :cite:`Giele:2002hx`.

            Parameters
            ----------
                path : string
                    output path
                ret : dict
                    DGLAP result
                ref : dict
                    reference result
        """
        pp = PdfPages(path)
        # get
        my_pdfs, my_pdf_errs = dglap.apply_operator(ret, LHA_init_pdfs, toy_xgrid)
        # iterate all pdf
        for key in my_pdfs:
            # skip trivial plots
            if key in ["V8", "V15", "V24", "V35", "T35"]:
                continue
            # plot
            fig = plot_dist(
                toy_xgrid,
                toy_xgrid * my_pdfs[key],
                toy_xgrid * my_pdf_errs[key],
                ref[key],
                title=f"x{key}(x,µ_F^2 = 10^4 GeV^2)",
            )
            pp.savefig()
            plt.close(fig)
        # close
        pp.close()


def save_initial_scale_plots_to_pdf(path):
    """
        Plots all PDFs at the inital scale.

        The reference values are given in Table 2 part 1 of :cite:`Giele:2002hx`.

        This excercise was usfull in order to detect the missing 2 in the definition of
        :math:`L_+ = 2(\\bar u + \\bar d)`

        Parameters
        ----------
            path : string
                output path
    """
    pp = PdfPages(path)
    # iterate all raw labels
    for j, label in enumerate(raw_label_list):
        # skip trivial plots
        if label in ["c_+", "b_+"]:
            continue
        me = LHA_init_grid[j]
        ref = LHA_init_grid_ref[j]
        fig = plot_dist(
            toy_xgrid,
            toy_xgrid * me,
            np.zeros(len(me)),
            ref,
            title=f"x{label}(x,µ_F^2 = 2 GeV^2)",
        )
        pp.savefig()
        plt.close(fig)
    # close
    pp.close()


if __name__ == "__main__":
    # setup
    n_low = 30
    n_mid = 20
    polynom_rank = 4

    # combine grid
    flag = f"l{n_low}m{n_mid}r{polynom_rank}-p5"
    xgrid_low = interpolation.get_xgrid_linear_at_log(
        n_low, 1e-7, 1.0 if n_mid == 0 else 0.1
    )
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
    app = LHABenchmarkPaper(xgrid, polynom_rank, flag)
    # check input scale
    #save_initial_scale_plots_to_pdf(assets_path / f"LHA-LO-FFNS-init-{flag}.pdf")
    # check fixed flavours
    app.run_FFNS()
    # do two steps
    #app.run_FFNS_twostep(1e2)
    # check ZM-VFNS
#     app.run_ZMVFNS()
