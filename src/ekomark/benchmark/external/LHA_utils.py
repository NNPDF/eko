"""
Implementation of :cite:`Giele:2002hx` and  :cite:`Dittmar:2005ed` (NNLO)
"""
import pathlib

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eko import basis_rotation as br

from ...plots import plot_dist

here = pathlib.Path(__file__).parents[0]

# xgrid
toy_xgrid = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])

raw_label_list = ["u_v", "d_v", "L_m", "L_p", "s_p", "c_p", "b_p", "g"]

# rot_func_list = [toy_V0, toy_V30, toy_T30, toy_T80, toy_S0, toy_S0, toy_S0, toy_g0]

# my/exact initial grid
# LHA_init_grid = []
# for f in [toy_uv0, toy_dv0, toy_Lm0, toy_Lp0, toy_sp0, toy_cp0, toy_bp0, toy_g0]:
#    LHA_init_grid.append(f(toy_xgrid))
LHA_init_grid = np.array([])


# L+ = 2(ub + db) = u+ - u- + d+ - d-
# L- = db - ub = ((d+ - d-) - (u+-u-))/2
# In the NNLO paper :cite:`Dittmar:2005ed` L_+ definition is different:
# L_+ = 2 L_+_NNLO
LHA_rotate_to_flavor = np.array(
    [
        # u_v, d_v, L_-, L_+, s_+, c_+, b_+,   g
        [0, 0, 0, 0, 0, 0, 0, 0],  # ph
        [0, 0, 0, 0, 0, 0, 0, 0],  # tbar
        [0, 0, 0, 0, 0, 0, 1 / 2, 0],  # bbar
        [0, 0, 0, 0, 0, 1 / 2, 0, 0],  # cbar
        [0, 0, 0, 0, 1 / 2, 0, 0, 0],  # sbar
        [0, 0, -1 / 2, 1 / 4, 0, 0, 0, 0],  # ubar
        [0, 0, 1 / 2, 1 / 4, 0, 0, 0, 0],  # dbar
        [0, 0, 0, 0, 0, 0, 0, 1],  # g
        [0, 1, 1 / 2, 1 / 4, 0, 0, 0, 0],  # d
        [1, 0, -1 / 2, 1 / 4, 0, 0, 0, 0],  # u
        [0, 0, 0, 0, 1 / 2, 0, 0, 0],  # s
        [0, 0, 0, 0, 0, 1 / 2, 0, 0],  # c
        [0, 0, 0, 0, 0, 0, 1 / 2, 0],  # b
        [0, 0, 0, 0, 0, 0, 0, 0],  # t
    ]
)

# rotate basis
def rotate_data(raw, is_ffns_nnlo=False, rotate_to_evolution_basis=False):
    """
    Rotate data either to flavor space or evolution space (from LHA space)
    which is yet an other basis.

    Parameters
    ----------
        raw : dict
            data
        is_ffns_nnlo : bool
            special table for NNLO FFNS
        rotate_to_evolution_basis : bool
            to evolution basis?

    Returns
    -------
        dict
            rotated data
    """
    inp = []
    label_list = raw_label_list
    to_flavor = LHA_rotate_to_flavor
    to_evolution = np.copy(br.rotate_flavor_to_evolution)
    if is_ffns_nnlo:
        # add s_v and delete b_p to label_list
        label_list = np.insert(label_list, 4, "s_v")
        label_list = np.delete(label_list, -2)

        # change the rotation matrix
        b_line = [0, 0, 0, 0, 0, 0, 1 / 2, 0]
        # b to zeros
        to_flavor = np.where(to_flavor == b_line, np.zeros(8), to_flavor)
        # c_p to b_p position
        to_flavor[3, :] = b_line
        to_flavor[-3, :] = b_line
        # s
        to_flavor[4, :] = [0, 0, 0, 0, -1 / 2, 1 / 2, 0, 0]
        to_flavor[-4, :] = [0, 0, 0, 0, 1 / 2, 1 / 2, 0, 0]

        # s_v = c_v count twice
        to_evolution[3, 4] = -2
        to_evolution[3, -4] = 2

    for l in label_list:
        inp.append(raw[l])
    inp = np.array(inp)

    flav_pdfs = np.dot(to_flavor, inp)

    # additional rotation to evolution basis if necessary
    if rotate_to_evolution_basis:
        evol_pdfs = np.matmul(to_evolution, flav_pdfs)
        return dict(zip(br.evol_basis, evol_pdfs))
    return dict(zip(br.flavor_basis_pids, flav_pdfs))


def compute_LHA_data(
    theory, operators, rotate_to_evolution_basis=False
):  # pylint: disable=too-many-statements disable=too-many-branches
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

    Q2grid = operators["Q2grid"]
    if not np.allclose(Q2grid, [1e4]):
        raise ValueError("Q2grid has to be [1e4]")
    # load data
    with open(here / "LHA.yaml", encoding="utf-8") as o:
        data = yaml.safe_load(o)

    fns = theory["FNS"]
    order = theory["PTO"]
    xif = (theory["fact_to_ren_scale_ratio"]) ** 2
    if order == 0 and xif != 1.0:
        raise ValueError("LO LHA tables with scale variations are not available")
    table = None
    part = None
    is_ffns_nnlo = False

    # Switching at the intermediate point.
    if xif > np.sqrt(2):
        part = 3
    elif xif < np.sqrt(1.0 / 2.0):
        part = 2
    else:
        part = 1

    if fns == "FFNS":
        if order == 0:
            table = 2
            part = 2
        elif order == 1:
            table = 3
        elif order == 2:
            is_ffns_nnlo = True
            table = 14
    elif fns == "ZM-VFNS":
        if order == 0:
            table = 2
            part = 3
        elif order == 1:
            table = 4
        elif order == 2:
            table = 15

    else:
        raise ValueError(f"unknown FNS {fns} or order {order}")
    ref_values = rotate_data(
        data[f"table{table}"][f"part{part}"], is_ffns_nnlo, rotate_to_evolution_basis
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
        path : str
        output path
    """
    # load data
    with open(here / "LHA.yaml", encoding="utf-8") as o:
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
