"""LHAPDF interface."""

import numpy as np

from eko import basis_rotation as br


def compute_LHAPDF_data(
    theory, operators, pdf, skip_pdfs, rotate_to_evolution_basis=False
):
    """Run LHAPDF to compute operators.

    Parameters
    ----------
    operators : dict
        operators card
    pdf : lhapdf_type
        pdf
    skip_pdfs : list
        list of pdfs (pid or name) to skip
    rotate_to_evolution_basis: bool
        rotate to evolution basis

    Returns
    -------
    ref : dict
        output containing: target_xgrid, values
    """
    target_xgrid = operators["interpolation_xgrid"]

    out_tabs = {}
    for mu2 in np.array(operators["mugrid"]) ** 2:
        tab = {}
        for pid in br.flavor_basis_pids:
            if pid in skip_pdfs:
                continue

            # collect lhapdf
            me = []
            for x in target_xgrid:
                xf = pdf.xfxQ2(pid, x, mu2)
                me.append(xf)
            tab[pid] = np.array(me)

        # rotate if needed
        if rotate_to_evolution_basis:
            qed = theory["QED"] > 0
            if not qed:
                evol_basis = br.evol_basis
                rotate_flavor_to_evolution = br.rotate_flavor_to_evolution
            else:
                evol_basis = br.unified_evol_basis
                rotate_flavor_to_evolution = br.rotate_flavor_to_unified_evolution
            pdfs = np.array(
                [
                    tab[pid] if pid in tab else np.zeros(len(target_xgrid))
                    for pid in br.flavor_basis_pids
                ]
            )
            evol_pdf = rotate_flavor_to_evolution @ pdfs
            tab = dict(zip(evol_basis, evol_pdf))

        out_tabs[mu2] = tab

    ref = {
        "target_xgrid": target_xgrid,
        "values": out_tabs,
    }

    return ref
