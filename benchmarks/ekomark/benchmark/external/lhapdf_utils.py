# -*- coding: utf-8 -*-
import numpy as np

import eko
from eko import basis_rotation as br


def compute_LHAPDF_data(theory, operators, pdf, rotate_to_evolution_basis=False):
    """
    Run LHAPDF

    Parameters
    ----------
        theory : dict
            theory card
        operators : dict
            operators card
        pdf : lhapdf_type
            pdf 
        rotate_to_evolution_basis: bool 
            rotate to evolution basis
    
    Returns
    -------
        ref : dict
            output containing: target_xgrid, values, pdf settings  
    """

    target_xgrid = operators["interpolation_xgrid"]
    src_pdf = pdf.set().name
    skip_pdfs = [22, -6, -5, 5, 6]

    out_tabs = {}
    for pid in br.flavor_basis_pids:

        # TODO: do we  need it??
        if pid in skip_pdfs:
            continue

        # collect lhapdf
        me = []
        for x in target_xgrid:
            xf = pdf.xfxQ2(pid, x, operators["Q2grid"][0])
            me.append(xf)
        out_tabs[pid] = np.array(me)

    # rotate if needed
    if rotate_to_evolution_basis:
        pdfs = np.array(
            [
                out_tabs[pid] if pid in out_tabs else np.zeros(len(target_xgrid))
                for pid in br.flavor_basis_pids
            ]
        )
        evol_pdf = br.rotate_flavor_to_evolution @ pdfs
        out_tabs = dict(zip(br.evol_basis, evol_pdf))

    ref = {
        "target_xgrid": target_xgrid,
        "values": {operators["Q2grid"][0]: out_tabs},
        "src_pdf": src_pdf,
        "rotate_to_evolution_basis": rotate_to_evolution_basis,
        "skip_pdfs": skip_pdfs,
    }

    return ref
