# -*- coding: utf-8 -*-
"""
APFEL interface
"""
import time

import numpy as np
from banana.benchmark.external.apfel_utils import load_apfel

from eko import basis_rotation as br


def compute_apfel_data(
    theory, operators, pdf, skip_pdfs, rotate_to_evolution_basis=False
):

    """
    Run APFEL to compute operators.

    Parameters
    ----------
        theory : dict
            theory card
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

    target_xgrid = operators["xgrid"]
    pdf_name = pdf.set().name

    # Load apfel
    apf_start = time.perf_counter()
    if theory["ModEv"] in ["EXA", "perturbative-exact"]:
        theory["ModEv"] = "EXA"
    elif theory["ModEv"] in ["EXP", "decompose-expanded", "perturbative-expanded"]:
        theory["ModEv"] = "EXP"
    elif theory["ModEv"] in ["TRN", "ordered-truncated"]:
        theory["ModEv"] = "TRN"
    else:
        raise ValueError(f"Method {theory['ModEv']} is not recognized. ")
    apfel = load_apfel(theory, operators, pdf_name)

    # Truncated Epsilon
    # APFEL::SetEpsilonTruncation(1E-1);
    #
    # Set maximum scale
    # APFEL::SetQLimits(theory.Q0, theory.QM );
    #
    # if (theory.SIA)
    # {
    #   APFEL::SetPDFSet("kretzer");
    #   APFEL::SetTimeLikeEvolution(true);
    # }

    # Set APFEL interpolation grid
    #
    # apfel.SetNumberOfGrids(3)
    # apfel.SetGridParameters(1, 50, 3, 1e-5)
    # apfel.SetGridParameters(2, 50, 3, 2e-1)
    # apfel.SetGridParameters(3, 50, 3, 8e-1)

    # init evolution
    apfel.InitializeAPFEL()
    print(f"Loading APFEL took {(time.perf_counter() - apf_start)} s")

    # Run
    apf_tabs = {}
    for q2 in operators["Q2grid"]:

        apfel.EvolveAPFEL(theory["Q0"], np.sqrt(q2))
        print(f"Executing APFEL took {(time.perf_counter() - apf_start)} s")

        tab = {}
        for pid in br.flavor_basis_pids:

            if pid in skip_pdfs:
                continue

            # collect APFEL
            apf = []
            for x in target_xgrid:
                xf = apfel.xPDF(pid if pid != 21 else 0, x)
                # if pid == 4:
                #     print(pid,x,xf)
                apf.append(xf)
            tab[pid] = np.array(apf)

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

        apf_tabs[q2] = tab

    ref = {
        "target_xgrid": target_xgrid,
        "values": apf_tabs,
    }

    return ref
