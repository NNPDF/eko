# -*- coding: utf-8 -*-
"""
Implementation of :cite:`Giele:2002hx`
"""
import pathlib
import yaml
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eko import basis_rotation as br

from ...plots import plot_dist

here = pathlib.Path(__file__).parents[0]

# xgrid
toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

# list
raw_label_list = {
    0: ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"],
    1: ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"],
    2: ["u_v", "d_v", "L_m", "L_p", "s_v", "s_p", "c_p", "g"],
}
# rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

evol_label_list = {
    1: ["V", "V3", "T3", "T8", "T15", "T24", "S", "g"],
    2: ["V", "V3", "T3", "V8", "T8", "T15", "T24", "S", "g"],
}

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

LHA_rotate_to_evolution_NNLO = np.array([
    # u_v, d_v, L_-, L_+, s_v s_+, c_+,    g
    [   1,   1,   0,   0,   2,   0,   0,   0], # V
    [   1,  -1,   0,   0,   0,   0,   0,   0], # V3
    [   1,  -1,  -2,   0,   0,   0,   0,   0], # T3
    [   1,   1,   0,   0,  -2,   0,   0,   0], # V8
    [   1,   1,   0,   1,   0,  -2,   0,   0], # T8
    [   1,   1,   0,   1,   0,   1,  -3,   0], # T15
    [   1,   1,   0,   1,   0,   1,   1,   0], # T24
    [   1,   1,   0,   1,   0,   1,   1,   0], # S
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

LHA_rotate_to_flavor_NNLO = np.array([
    # u_v, d_v, L_-, L_+, s_v  s_+, c_+,   g
    [   0,   0,   0,   0,   0,   0,   0,   0], # ph
    [   0,   0,   0,   0,   0,   0,   0,   0], # tbar
    [   0,   0,   0,   0,   0,   0,   0,   0], # bbar
    [   0,   0,   0,   0,   0,   0, 1/2,   0], # cbar
    [   0,   0,   0,   0,-1/2, 1/2,   0,   0], # sbar
    [   0,   0, 1/2, 1/4,   0,   0,   0,   0], # ubar
    [   0,   0,-1/2, 1/4,   0,   0,   0,   0], # dbar
    [   0,   0,   0,   0,   0,   0,   0,   1], # g
    [   0,   1,-1/2, 1/4,   0,   0,   0,   0], # d
    [   1,   0, 1/2, 1/4,   0,   0,   0,   0], # u
    [   0,   0,   0,   0, 1/2, 1/2,   0,   0], # s
    [   0,   0,   0,   0,   0,   0, 1/2,   0], # c
    [   0,   0,   0,   0,   0,   0,   0,   0], # b
    [   0,   0,   0,   0,   0,   0,   0,   0], # t
])
# fmt: on

# rotate basis
def rotate_data(raw, order, rotate_to_evolution_basis=False):
    """
    Rotate data either to flavor space or evolution space (from LHA space)
    which is yet an other basis.

    Parameters
    ----------
        raw : dict
            data
        order : int
            perturbative order
        rotate_to_evolution_basis : bool
            to evolution basis?

    Returns
    -------
        dict
            rotated data
    """
    inp = []
    for l in raw_label_list[order]:
        inp.append(raw[l])
    inp = np.array(inp)
    if rotate_to_evolution_basis:
        if order == 2:
            rot = np.dot(LHA_rotate_to_evolution_NNLO, inp)
            return dict(zip(evol_label_list[2], rot))
        else:
            rot = np.dot(LHA_rotate_to_evolution, inp)
            return dict(zip(evol_label_list[1], rot))
    else:
        if order == 2:
            rot = np.dot(LHA_rotate_to_flavor_NNLO, inp)
        else:
            rot = np.dot(LHA_rotate_to_flavor, inp)
        return dict(zip(br.flavor_basis_pids, rot))


def compute_LHA_data(theory, operators, rotate_to_evolution_basis=False):
    """
    Setup LHA benchmark :cite:`Giele:2002hx`

    Parameters
    ----------
        theory : dict
            theory card
        operators : dict
            operators card
        rotate_to_evolution_basis : bool
            rotate to evolution basis

    Returns
    -------
        ref : dict
            output containing: target_xgrid, values
    """

    if not np.isclose(theory["XIF"], 1.0):
        raise ValueError("XIF has to be 1")
    Q2grid = operators["Q2grid"]
    if not np.allclose(Q2grid, [1e4]):
        raise ValueError("Q2grid has to be [1e4]")
    # load data
    with open(here / "LHA.yaml") as o:
        data = yaml.safe_load(o)

    fns = theory["FNS"]
    order = theory["PTO"]
    fact_to_ren = (theory["XIF"] / theory["XIR"]) ** 2
    table = None
    part = None
    if fns == "FFNS":
        if order == 0:
            table = 2
            part = 2
        elif order == 1:
            table = 3
            # Switching at the intermediate point.
            if fact_to_ren > np.sqrt(2):
                part = 3
            elif fact_to_ren < np.sqrt(1.0 / 2.0):
                part = 2
            else:
                part = 1
        elif order == 2:
            table = 14
            if fact_to_ren > np.sqrt(2):
                part = 3
            elif fact_to_ren < np.sqrt(1.0 / 2.0):
                part = 2
            else:
                part = 1
    elif fns == "ZM-VFNS":
        if order == 0:
            table = 2
            part = 3
        elif order == 1:
            table = 4
            if fact_to_ren > np.sqrt(2):
                part = 3
            elif fact_to_ren < np.sqrt(1.0 / 2.0):
                part = 2
            else:
                part = 1
        elif order == 2:
            table = 15
            if fact_to_ren > np.sqrt(2):
                part = 3
            elif fact_to_ren < np.sqrt(1.0 / 2.0):
                part = 2
            else:
                part = 1
    else:
        raise ValueError(f"unknown FNS {fns} or order {order}")
    ref_values = rotate_data(
        data[f"table{table}"][f"part{part}"], order, rotate_to_evolution_basis
    )
    ref = {
        "target_xgrid": toy_xgrid,
        "values": {1e4: ref_values},
    }

    return ref


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
    # load data
    with open(here / "LHA.yaml") as o:
        data = yaml.safe_load(o)
    LHA_init_grid_ref = data["table2"]["part1"]
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
