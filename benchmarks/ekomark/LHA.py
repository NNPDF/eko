# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .plots import plot_dist
from .runner import Runner
from eko.basis_rotation import flavor_basis_pids

# xgrid
toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

# list
raw_label_list = ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"]
# rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

# my/exact initial grid
# LHA_init_grid = []
# for f in [toy_uv0, toy_dv0, toy_Lm0, toy_Lp0, toy_sp0, toy_cp0, toy_bp0, toy_g0]:
#    LHA_init_grid.append(f(toy_xgrid))
LHA_init_grid = np.array([])

# fmt: off
# rotation matrix
# LHA_flavour_rotate = np.array([
#     # u_v, d_v, L_-, L_+, s_+, c_+, b_+,   g
#     [   1,   1,   0,   0,   0,   0,   0,   0], # V
#     [   1,  -1,   0,   0,   0,   0,   0,   0], # V3
#     [   1,  -1,  -2,   0,   0,   0,   0,   0], # T3
#     [   1,   1,   0,   1,  -2,   0,   0,   0], # T8
#     [   1,   1,   0,   1,   1,  -3,   0,   0], # T15
#     [   1,   1,   0,   1,   1,   1,  -4,   0], # T24
#     [   1,   1,   0,   1,   1,   1,   1,   0], # S
#     [   0,   0,   0,   0,   0,   0,   0,   1], # g
# ])
LHA_flavour_rotate = np.array([
    # u_v, d_v, L_-, L_+, s_+, c_+, b_+,   g
    [   0,   0,   0,   0,   0,   0,   0,   0], # ph
    [   0,   0,   0,   0,   0,   0,   0,   0], # tbar
    [   0,   0,   0,   0,   0,   0,   1,   0], # bbar
    [   0,   0,   0,   0,   0,   1,   0,   0], # cbar
    [   0,   0,   0,   0,   1,   0,   0,   0], # sbar
    [  -1,   0,  -1,   2,   0,   0,   0,   0], # ubar
    [   0,  -1,   1,   2,   0,   0,   0,   0], # dbar
    [   0,   0,   0,   0,   0,   0,   0,   1], # g
    [   0,   1,   0,   0,   0,   0,   0,   0], # d
    [   1,   0,   0,   0,   0,   0,   0,   0], # u
    [   0,   0,   0,   0,   1,   0,   0,   0], # s
    [   0,   0,   0,   0,   0,   1,   0,   0], # c
    [   0,   0,   0,   0,   0,   0,   1,   0], # b
    [   0,   0,   0,   0,   0,   0,   0,   0], # t
])
# fmt: on

# rotate basis
def rotate_data(raw):
    inp = []
    for l in raw_label_list:
        inp.append(raw[l])
    inp = np.array(inp)
    rot = np.dot(LHA_flavour_rotate, inp)
    return dict(zip(flavor_basis_pids, rot))


class LHABenchmarkPaper(Runner):
    """
    Compares to the LHA benchmark paper :cite:`Giele:2002hx`.

    Parameters
    ----------
        path : string or pathlib.Path
            path to input card
        data_dir : string
            data directory
        assets_dir : string
            output directory
    """

    def __init__(self, theory_path, operators_path, assets_dir, data_dir):
        super().__init__(theory_path, operators_path, assets_dir)

        if not np.isclose(self.theory["XIF"], 1.0):
            raise ValueError("XIF has to be 1")
        Q2grid = self.operators["Q2grid"]
        if not np.allclose(Q2grid, [1e4]):
            raise ValueError("Q2grid has to be [1e4]")
        # load data
        with open(data_dir / "LHA.yaml") as o:
            self.data = yaml.safe_load(o)

    def ref_values(self):
        """
        Load the reference data from the paper.

        Returns
        -------
            ref : dict
                (rotated) reference data
        """
        fns = self.theory["FNS"]
        order = self.theory["PTO"]
        fact_to_ren = (self.theory["XIF"] / self.theory["XIR"]) ** 2
        if fns == "FFNS":
            if order == 0:
                return rotate_data(self.data["table2"]["part2"])
            if order == 1:
                if fact_to_ren > np.sqrt(2):
                    return rotate_data(self.data["table3"]["part3"])
                if fact_to_ren < np.sqrt(1.0 / 2.0):
                    return rotate_data(self.data["table3"]["part2"])
                return rotate_data(self.data["table3"]["part1"])
        if fns == "ZM-VFNS":
            if order == 0:
                return rotate_data(self.data["table2"]["part3"])
            if order == 1:
                if fact_to_ren > np.sqrt(2):
                    return rotate_data(self.data["table4"]["part3"])
                if fact_to_ren < np.sqrt(1.0 / 2.0):
                    return rotate_data(self.data["table4"]["part2"])
                return rotate_data(self.data["table4"]["part1"])
        raise ValueError(f"unknown FNS {fns} or order {order}")

    def ref(self):
        skip_pdfs = [22, -6, 6]
        if self.theory["FNS"] == "FFNS":
            skip_pdfs.extend([-5, 5])
        return {
            "target_xgrid": toy_xgrid,
            "values": {1e4: self.ref_values()},
            "src_pdf": "ToyLH",
            "is_flavor_basis": False,
            "skip_pdfs": skip_pdfs,
        }

    def save_initial_scale_plots_to_pdf(self, path):
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
        LHA_init_grid_ref = self.data["table2"]["part1"]
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
