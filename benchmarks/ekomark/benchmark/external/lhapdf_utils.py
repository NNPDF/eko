# -*- coding: utf-8 -*-
"""
LHAPDF interface
"""
import numpy as np

from eko import basis_rotation as br


def compute_LHAPDF_data(operators, pdf, skip_pdfs, Q2s=None, rotate_to_evolution_basis=False):
    """
    Run LHAPDF to compute operators.

    Parameters
    ----------
        operators : dict
            operators card
        pdf : lhapdf_type
            PDF
        skip_pdfs : list
            list of pdfs (pid or name) to skip
        Q2s : list(float)
            compute the pdf at the given q2, otherwise use the Q2grid values
        rotate_to_evolution_basis: bool
            rotate to evolution basis

    Returns
    -------
        ref : dict
            output containing: target_xgrid, values
    """

    target_xgrid = operators["interpolation_xgrid"]
    out_tabs = {}

    if Q2s is None:
        Q2s = operators["Q2grid"]
    # loop on q2
    for q2 in Q2s:
        tab = {}

        # loop on particles
        for pid in br.flavor_basis_pids:

            if pid in skip_pdfs:
                continue

            # collect lhapdf
            tab[pid] = np.array([pdf.xfxQ2(pid, x, q2) for x in target_xgrid])

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

    ref = {
        "target_xgrid": target_xgrid,
        "values": out_tabs,
    }

    return ref
