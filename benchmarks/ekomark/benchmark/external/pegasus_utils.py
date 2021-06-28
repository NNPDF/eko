# -*- coding: utf-8 -*-
"""
Pegasus interface
"""
import numpy as np

from eko import basis_rotation as br


def compute_pegasus_data(theory, operators, skip_pdfs, rotate_to_evolution_basis=False):
    """
    Run Pegasus to compute operators.

    Parameters
    ----------
        theory : dict
            theory card
        operators : dict
            operators card
        skip_pdfs : list
            list of pdfs (pid or name) to skip
        rotate_to_evolution_basis: bool
            rotate to evolution basis

    Returns
    -------
        ref : dict
            output containing: target_xgrid, values
    """
    import pegasus  # pylint:disable=import-error,import-outside-toplevel

    target_xgrid = operators["interpolation_xgrid"]

    # init pegasus
    nf = theory["NfFF"]

    if theory["ModEv"] in ["EXA", "perturbative-exact"]:
        imodev = 1
    elif theory["ModEv"] in ["EXP", "decompose-expanded", "perturbative-expanded"]:
        imodev = 2
    elif theory["ModEv"] in ["ordered-truncated"]:
        imodev = 3
    elif theory["ModEv"] in ["TRN"]:
        imodev = 4  # 4 is random - just not 1-3
    else:
        raise ValueError(f"Method {theory['ModEv']} is not recognized. ")

    ivfns = 0 if theory["FNS"] == "FFNS" else 1

    if theory["Q0"] != theory["Qref"]:
        raise ValueError("Initial scale Q0 must be equal to Qref in Pegasus.")

    pegasus.initevol(
        imodev, theory["PTO"], ivfns, nf, theory["fact_to_ren_scale_ratio"] ** 2
    )
    pegasus.initinp(
        theory["alphas"],
        theory["Qref"] ** 2,
        (theory["kcThr"] * theory["mc"]) ** 2,
        (theory["kbThr"] * theory["mb"]) ** 2,
        (theory["ktThr"] * theory["mt"]) ** 2,
    )

    # better return always the flavor basis and then rotate
    # if rotate_to_evolution_basis:
    #     ipstd = 0
    #     labels = list(br.evol_basis)
    # else:
    #     ipstd = 1

    # photon pdf is not in pagsus output
    labels = list(br.flavor_basis_pids)
    labels.remove(22)
    # swap u, d and ubar and dbar
    labels[4], labels[5] = labels[5], labels[4]
    labels[-6], labels[-5] = labels[-5], labels[-6]

    # run pegaus
    out_tabs = {}
    for q2 in operators["Q2grid"]:

        tab = {}
        for x in target_xgrid:
            # last two numbers are the min and max pid to calculate,
            # keep everthing for simplicity.
            xf, _ = pegasus.xparton(x, q2, -6, 6)
            temp = dict(zip(labels, xf))
            for pid in labels:
                if pid in skip_pdfs:
                    continue
                if pid not in tab.keys():
                    tab[pid] = []
                tab[pid].append(temp[pid])

        # rotate if needed
        if rotate_to_evolution_basis:
            pdfs = np.array(
                [
                    tab[pid] if pid in tab else np.zeros(len(target_xgrid))
                    for pid in br.flavor_basis_pids
                ]
            )
            evol_pdf = br.rotate_flavor_to_evolution @ pdfs
            tab = dict(zip(br.evol_basis, evol_pdf))

        out_tabs[q2] = tab

    ref = {"target_xgrid": target_xgrid, "values": out_tabs}

    return ref
