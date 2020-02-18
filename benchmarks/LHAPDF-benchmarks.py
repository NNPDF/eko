# -*- coding: utf-8 -*-
"""
    Benchmark to LHAPDF
"""
import logging
import sys
import os
import copy
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lhapdf
import yaml

import eko.dglap as dglap
import eko.interpolation as interpolation
import eko.utils as utils

from tools import plot_dist, save_all_operators_to_pdf


# output path
assets_path = pathlib.Path(__file__).with_name("assets")
lhapdf.pathsAppend("/usr/local/share/LHAPDF/")


class LHAPDFBenchmark:
    """
        Compares to LHAPDF.

        Parameters
        ----------
            pdfname : array
                pdf name
            pdfmember : int
                PDF member
            xgrid : array
                basis grid
            polynomial_degree : int
                degree of interpolation polynomial
            flag : string
                file name tag
    """

    def __init__(self, pdfname, pdfmember, xgrid, polynomial_degree, flag):
        self._pdfname = pdfname
        self._pdfmember = pdfmember
        self._pdfset = lhapdf.getPDFSet(self._pdfname)
        self._pdf = self._pdfset.mkPDF(self._pdfmember)
        self._xgrid = xgrid
        self._polynomial_degree = polynomial_degree
        self._flag = flag
        # default config for post processing
        self.post_process_config = {
            "plot_PDF": True,
            "plot_operator": True,
            "write_operator": True,
        }
        self._operator_path = (
            assets_path
            / f"LHAPDF-{self._pdfname}-{self._pdfmember}-ops-{self._flag}.yaml"
        )
        # default setup
        if self._pdfset.get_entry("OrderQCD") != "0":
            raise NotImplementedError("Currently only LO is supported!")
        self._setup = {
            "PTO": 0,
            "alphas": float(self._pdfset.get_entry("AlphaS_MZ")),
            "Qref": float(self._pdfset.get_entry("MZ")),
            "FNS": "FFNS",
            "NfFF": int(self._pdfset.get_entry("NumFlavors")),
            "xgrid_type": "custom",
            "xgrid": xgrid,
            "xgrid_polynom_rank": polynom_rank,
        }
        self._Q2init = None
        self._Q2final = None

    def _post_process(self, ret):
        """
            Handles the post processing of the run

            Parameters
            ----------
                ret : dict
                    DGLAP result
        """
        if self.post_process_config["plot_PDF"]:
            self._save_final_scale_plots_to_pdf(
                assets_path
                / f"LHAPDF-{self._pdfname}-{self._pdfmember}-plots-{self._flag}.pdf",
                ret,
            )
        if self.post_process_config["plot_operator"]:
            save_all_operators_to_pdf(
                ret,
                assets_path
                / f"LHAPDF-{self._pdfname}-{self._pdfmember}-ops-{self._flag}.pdf",
            )
        if self.post_process_config["write_operator"]:
            dglap.write_YAML_to_file(ret, self._operator_path)

    def run_FFNS(self, Q2init, Q2final):
        """
            Runs a fixed-flavor-number-scheme configuration.

            Parameters
            ----------
                Q2init : t_flaot
                    init scale squared
                Q2final : t_flaot
                    final scale squared

            Returns
            -------
                ret : dict
                    DGLAP result
        """
        self._Q2init = Q2init
        self._Q2final = Q2final
        if os.path.exists(self._operator_path):
            # read
            with open(self._operator_path) as o:
                ret_raw = yaml.load(o, Loader=yaml.BaseLoader)
            # remap
            ret = {"operators": {}, "operator_errors": {}}
            ret["xgrid"] = np.array(ret_raw["xgrid"], float)
            ret["polynomial_degree"] = int(ret_raw["polynomial_degree"])
            ret["log"] = bool(ret_raw["log"])
            ret["basis"] = interpolation.InterpolatorDispatcher(ret["xgrid"],ret["polynomial_degree"],ret["log"])
            for k in ret_raw["operators"]:
                ret["operators"][k] = np.array(ret_raw["operators"][k], float)
                ret["operator_errors"][k] = np.array(
                    ret_raw["operator_errors"][k], float
                )
        else:
            add_setup = {"Q0": np.sqrt(self._Q2init), "Q2grid": [self._Q2final]}
            # xgrid can be a copy, so we don't need a deep copy here
            setup = utils.merge_dicts(copy.copy(self._setup), add_setup)
            ret = dglap.run_dglap(setup)
        self._post_process(ret)
        return ret

    def _save_final_scale_plots_to_pdf(self, path, ret):
        """
            Plots all PDFs at the final scale.

            Parameters
            ----------
                path : string
                    output path
                ret : dict
                    DGLAP result
        """

        # fmt: off
        flavor_rotate = {}
        flavor_rotate["V"] = np.array(
            [[(k, 1), (-k, -1)] for k in range(1, 6 + 1)]
        ).reshape(-1, 2)  # = d_v + u_v + ...
        flavor_rotate["V3"]  = [(+1, -1),(-1, +1),(+2, +1),(-2, -1)]  # = - d_v + u_v
        flavor_rotate["T3"]  = [(+1, -1),(-1, -1),(+2, +1),(-2, +1)]  # = - d_p + u_p
        flavor_rotate["T8"]  = [(+1, +1),(-1, +1),(+2, +1),(-2, +1),(+3, -2),(-3, -2),]  # = d_p + u_p - 2s_p
        flavor_rotate["T15"] = [(+1, +1),(-1, +1),(+2, +1),(-2, +1),(+3, +1),(-3, +1),(+4, -3),(-4, -3),]  # = d_p + u_p + s_p - 3c_p
        flavor_rotate["T24"] = [(+1, +1),(-1, +1),(+2, +1),(-2, +1),(+3, +1),(-3, +1),(+4, +1),(-4, +1),(+5, -4),(-5, -4),]  # = d_p + u_p + s_p + c_p - 4b_p
        flavor_rotate["S"] = np.array(
            [[(k, 1), (-k, 1)] for k in range(1, 6 + 1)]
        ).reshape(-1, 2)  # = d + dbar + u + ubar + ...
        flavor_rotate["g"] = [(21, 1)]  # = g
        # fmt: on

        def get_get_evolution_member(label, mu2):
            def get_member(x):
                ret = 0
                for pid, c in flavor_rotate[label]:
                    if not self._pdf.hasFlavor(pid):
                        continue
                    ret += c * self._pdf.xfxQ2(pid, x, mu2) / x
                return ret

            return get_member

        # set init
        init_pdfs = {}
        for k in flavor_rotate:
            init_pdfs[k] = get_get_evolution_member(k, self._Q2init)

        target_grid = []
        for x in ret["xgrid"]:
            if x > 1e-5:
                target_grid.append(x)
        target_grid = np.array(target_grid)
        # set final
        ref = {}
        for k in flavor_rotate:
            pdf = get_get_evolution_member(k, self._Q2final)
            ref[k] = np.array([pdf(x) for x in target_grid])

        # evolve
        my_pdfs, my_pdf_errs = dglap.apply_operator(ret, init_pdfs, target_grid)

        #xs = np.logspace(-6,-4,20)
        #p1,pe1 = dglap.apply_operator(ret, init_pdfs,xs)
        #print(xs*p1["V"])
        #print(pe1)
        #raise "Blub"
        # iterate all pdf
        pp = PdfPages(path)
        for key in my_pdfs:
            # skip trivial plots
            if key in ["V8", "V15", "V24", "V35", "T35"]:
                continue
            # plot
            fig = plot_dist(
                target_grid,
                target_grid * my_pdfs[key],
                target_grid * my_pdf_errs[key],
                target_grid * ref[key],
                title=f"x{key}(x,µ_F^2 = {self._Q2final:g} GeV^2)",
            )
            pp.savefig()
            plt.close(fig)
        # close
        pp.close()


if __name__ == "__main__":
    # setup
    n_low = 40
    n_mid = 30
    polynom_rank = 4

    # combine grid
    flag = f"l{n_low}m{n_mid}r{polynom_rank}"
    xgrid_low = interpolation.get_xgrid_linear_at_log(
        n_low, 1e-8, 1.0 if n_mid == 0 else 0.1
    )
    xgrid_mid = interpolation.get_xgrid_linear_at_id(n_mid, 0.1, 1.0)
    xgrid = np.unique(np.concatenate((xgrid_low, xgrid_mid)))

    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko.dglap").handlers = []
    logging.getLogger("eko.dglap").addHandler(logStdout)
    logging.getLogger("eko.dglap").setLevel(logging.DEBUG)

    # run
    app = LHAPDFBenchmark("NNPDF31_lo_as_0118", 0, xgrid, polynom_rank, flag)
    app.run_FFNS(25, 1e4)
