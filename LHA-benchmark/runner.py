# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""
import logging
import sys
import copy
import pathlib
import pprint
import io

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plots import plot_dist, plot_operator

from eko import run_dglap

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


def toy_cp0(x):
    return 0 * x


def toy_bp0(x):
    return 0 * x


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
raw_label_list = ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"]
rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

# my/exact initial grid
LHA_init_grid = []
for f in [toy_uv0, toy_dv0, toy_Lm0, toy_Lp0, toy_sp0, toy_cp0, toy_bp0, toy_g0]:
    LHA_init_grid.append(f(toy_xgrid))
LHA_init_grid = np.array(LHA_init_grid)

# fmt: off
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
def rotate_data(raw):
    inp = []
    for l in raw_label_list:
        inp.append(raw[l])
    inp = np.array(inp)
    rot = np.dot(LHA_flavour_rotate, inp)
    out = {}
    for k, n in enumerate(["V", "V3", "T3", "T8", "T15", "T24", "S", "g"]):
        out[n] = rot[k]
    return out


# output path
here = pathlib.Path(__file__).parent
assets_path = here / "assets"

# load data
with open(here / "data.yaml") as o:
    LHA_data = yaml.safe_load(o)


class LHABenchmarkPaper:
    """
        Compares to the LHA benchmark paper :cite:`Giele:2002hx`.

        Parameters
        ----------
            order : int
                order of perturbation theory
            xgrid : array
                basis grid
            polynomial_degree : int
                degree of interpolation polynomial
            flag : string
                output file tag
            debug_skip_singlet : bool
                skip singlet integration
            debug_skip_non_singlet : bool
                skip singlet integration
    """

    def __init__(self, path):
        print(path)
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self.path = path
        with open(path, "r") as infile:
            self.setup = yaml.safe_load(infile)
        # default config for post processing
        self.post_process_config = {
            "plot_PDF": True,
            "plot_operator": True,
            "write_operator": True,
        }

    def _post_process(self, output):
        """
            Handles the post processing of the run according to the configuration.

            Parameters
            ----------
                output : eko.output.Output
                    EKO result
                ref : dict
                    reference result
                tag : string
                    file tag
        """
        # dump operators to file
        if self.post_process_config["write_operator"]:
            p = assets_path / (self.path.stem + "-ops.yaml")
            output.dump_yaml_to_file(p)
            print(f"write operator to {p}")
        # pdf comparison
        if self.post_process_config["plot_PDF"]:
            p = assets_path / (self.path.stem + "-plots.pdf")
            self.save_final_scale_plots_to_pdf(p, output)
            print(f"write pdf plots to {p}")
        # graphical representation of operators
        if self.post_process_config["plot_operator"]:
            p = assets_path / (self.path.stem + "-ops.pdf")
            self.save_all_operators_to_pdf(output, p)
            print(f"write operator plots to {p}")

    def ref(self):
        """
            Load the reference data from the paper.

            Parameters
            ----------
                fns : str
                    flavor number scheme

            Returns
            -------
                ref : dict
                    (rotated) reference data
        """
        fns = self.setup["FNS"]
        order = self.setup["PTO"]
        if fns == "FFNS":
            if order == 0:
                return rotate_data(LHA_data["table2"]["part2"])
            if order == 1:
                return rotate_data(LHA_data["table3"]["part1"])
        if fns == "ZM-VFNS":
            if order == 0:
                return rotate_data(LHA_data["table2"]["part3"])
            if order == 1:
                return rotate_data(LHA_data["table4"]["part1"])
        raise ValueError(f"unknown FNS {fns} or order {order}")

    def run(self):
        """
            Runs the input card

            Returns
            -------
                ret : dict
                    DGLAP result
        """
        ret = run_dglap(self.setup)
        self._post_process(ret)
        return ret

    def input_figure(self, output):
        """
            Pretty-prints the setup to a figure

            Parameters
            ----------
                output : eko.output.Output
                    DGLAP result

            Returns
            -------
                firstPage : matplotlib.pyplot.Figure
                    figure
        """
        firstPage = plt.figure(figsize=(10,8))
        firstPage.text(0.,1, "Setup:", size=14, ha="left",va="top")
        setup = copy.deepcopy(output)
        del setup["Q2grid"]
        str_stream = io.StringIO()
        pprint.pprint(setup, stream=str_stream)
        firstPage.text(0.,.95, str_stream.getvalue(), size=12, ha="left",va="top")
        return firstPage

    def save_all_operators_to_pdf(self, output, path):
        """
            Output all operator heatmaps to PDF.

            Parameters
            ----------
                output : eko.output.Output
                    DGLAP result
                path : string
                    target file name
        """
        first_ops = list(output["Q2grid"].values())[0]
        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(output)
            pp.savefig()
            plt.close(firstPage)
            # print operators
            for label in first_ops["operators"]:
                try:
                    fig = plot_operator(first_ops, label)
                    pp.savefig()
                finally:
                    if fig:
                        plt.close(fig)

    def save_final_scale_plots_to_pdf(self, path, output):
        """
            Plots all PDFs at the final scale.

            The reference values by :cite:`Giele:2002hx`.

            Parameters
            ----------
                path : string
                    output path
                output : eko.output.Output
                    DGLAP result
        """
        # get
        pdf_grid = output.apply_pdf(LHA_init_pdfs, toy_xgrid)
        first_res = list(pdf_grid.values())[0]
        my_pdfs = first_res["pdfs"]
        my_pdf_errs = first_res["errors"]
        ref = self.ref()
        with PdfPages(path) as pp:
            # print setup
            firstPage = self.input_figure(output)
            pp.savefig()
            plt.close(firstPage)
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
    LHA_init_grid_ref = LHA_data["table2"]["part1"]
    with PdfPages(path) as pp:
        # iterate all raw labels
        for j, label in enumerate(raw_label_list):
            # skip trivial plots
            if label in ["c_p", "b_p"]:
                continue
            me = LHA_init_grid[j]
            ref = LHA_init_grid_ref[label]
            fig = plot_dist(
                toy_xgrid,
                toy_xgrid * me,
                np.zeros(len(me)),
                ref,
                title=f"x{label}(x,µ_F^2 = 2 GeV^2)",
            )
            pp.savefig()
            plt.close(fig)
    print(f"Initial scale pdf plots written to {path}")


if __name__ == "__main__" and True:
    # activate logging
    logStdout = logging.StreamHandler(sys.stdout)
    logStdout.setLevel(logging.INFO)
    logStdout.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("eko").handlers = []
    logging.getLogger("eko").addHandler(logStdout)
    logging.getLogger("eko").setLevel(logging.INFO)

    # run
    if len(sys.argv) == 2:
        app = LHABenchmarkPaper(sys.argv[1])
        app.run()
    else:
        me = sys.argv[0]
        print(f"Usage: {me} path/to/input/card.yaml")

if __name__ == "__main__" and False:
    raw = """"""
    for r in raw:
        r = (
            r.replace("∗\n", "")
            .replace(" −", "e-")
            .replace(" +", "e+")
            .replace("\n", ",")
        )
        print("[" + r + "]")
