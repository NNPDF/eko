# -*- coding: utf-8 -*-
"""
    Benchmark EKO to :cite:`Giele:2002hx`
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eko import basis_rotation as br

from .plots import plot_dist
from .runner import Runner

# xgrid
toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

# list
raw_label_list = ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"]
# rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

evol_label_list = ["V", "V3", "T3", "T8", "T15", "T24", "S", "g"]

# my/exact initial grid
# LHA_init_grid = []
# for f in [toy_uv0, toy_dv0, toy_Lm0, toy_Lp0, toy_sp0, toy_cp0, toy_bp0, toy_g0]:
#    LHA_init_grid.append(f(toy_xgrid))
LHA_init_grid = np.array([])

# fmt: off
# rotation matrix
LHA_rotate_to_evolution = np.array([
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
# L+ = 2(ub + db) = u+ - u- + d+ - d-
# L- = ub - db = ((u+-u-) - (d+ - d-))/2
LHA_rotate_to_flavor = np.array([
    # u_v, d_v, L_-, L_+, s_+, c_+, b_+,   g
    [   0,   0,   0,   0,   0,   0,   0,   0], # ph
    [   0,   0,   0,   0,   0,   0,   0,   0], # tbar
    [   0,   0,   0,   0,   0,   0, 1/2,   0], # bbar
    [   0,   0,   0,   0,   0, 1/2,   0,   0], # cbar
    [   0,   0,   0,   0, 1/2,   0,   0,   0], # sbar
    [   0,   0, 1/2, 1/4,   0,   0,   0,   0], # ubar
    [   0,   0,-1/2, 1/4,   0,   0,   0,   0], # dbar
    [   0,   0,   0,   0,   0,   0,   0,   1], # g
    [   0,   1,-1/2, 1/4,   0,   0,   0,   0], # d
    [   1,   0, 1/2, 1/4,   0,   0,   0,   0], # u
    [   0,   0,   0,   0, 1/2,   0,   0,   0], # s
    [   0,   0,   0,   0,   0, 1/2,   0,   0], # c
    [   0,   0,   0,   0,   0,   0, 1/2,   0], # b
    [   0,   0,   0,   0,   0,   0,   0,   0], # t
])
# fmt: on

# rotate basis
def rotate_data(raw, rotate_to_evolution_basis=False):
    inp = []
    for l in raw_label_list:
        inp.append(raw[l])
    inp = np.array(inp)
    if rotate_to_evolution_basis:
        rot = np.dot(LHA_rotate_to_evolution, inp)
        return dict(zip(evol_label_list, rot))
    else:
        rot = np.dot(LHA_rotate_to_flavor, inp)
        return dict(zip(br.flavor_basis_pids, rot))


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

        self.rotate_to_evolution_basis = False  # True

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
                return rotate_data(
                    self.data["table2"]["part2"], self.rotate_to_evolution_basis
                )
            if order == 1:
                if fact_to_ren > np.sqrt(2):
                    return rotate_data(
                        self.data["table3"]["part3"], self.rotate_to_evolution_basis
                    )
                if fact_to_ren < np.sqrt(1.0 / 2.0):
                    return rotate_data(
                        self.data["table3"]["part2"], self.rotate_to_evolution_basis
                    )
                return rotate_data(
                    self.data["table3"]["part1"], self.rotate_to_evolution_basis
                )
        if fns == "ZM-VFNS":
            if order == 0:
                return rotate_data(
                    self.data["table2"]["part3"], self.rotate_to_evolution_basis
                )
            if order == 1:
                if fact_to_ren > np.sqrt(2):
                    return rotate_data(
                        self.data["table4"]["part3"], self.rotate_to_evolution_basis
                    )
                if fact_to_ren < np.sqrt(1.0 / 2.0):
                    return rotate_data(
                        self.data["table4"]["part2"], self.rotate_to_evolution_basis
                    )
                return rotate_data(
                    self.data["table4"]["part1"], self.rotate_to_evolution_basis
                )
        raise ValueError(f"unknown FNS {fns} or order {order}")

    def ref(self):
        """
        Reference configuration
        """
        skip_pdfs = [22, -6, 6, "ph", "V35", "V24", "V15", "V8", "T35"]
        if self.theory["FNS"] == "FFNS":
            skip_pdfs.extend([-5, 5, "T24"])
        return {
            "target_xgrid": toy_xgrid,
            "values": {1e4: self.ref_values()},
            "src_pdf": "ToyLH",
            "skip_pdfs": skip_pdfs,
            "rotate_to_evolution_basis": self.rotate_to_evolution_basis,
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
