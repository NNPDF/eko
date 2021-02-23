# -*- coding: utf-8 -*-
import platform
import time
import numpy as np

from eko import basis_rotation as br

import apfel


def load_apfel(theory, operators, pdf="ToyLH"):
    """
    Set APFEL parameter from ``theory`` dictionary.

    Parameters
    ----------
    theory : dict
        theory card
    operators : dict
        operators card
    pdf : str
        PDF name 
    Returns
    -------
    module
        loaded apfel wrapper
    """

    # Cleanup APFEL common blocks
    apfel.CleanUp()

    # Theory, perturbative order of evolution
    if not theory.get("QED"):
        apfel.SetTheory("QCD")
    else:
        apfel.SetTheory("QUniD")
        apfel.EnableNLOQEDCorrections(True)
    apfel.SetPerturbativeOrder(theory.get("PTO"))

    if theory.get("ModEv") == "EXA":
        apfel.SetPDFEvolution("exactalpha")
        apfel.SetAlphaEvolution("exact")
    elif theory.get("ModEv") == "EXP":
        apfel.SetPDFEvolution("expandalpha")
        apfel.SetAlphaEvolution("expanded")
    elif theory.get("ModEv") == "TRN":
        apfel.SetPDFEvolution("truncated")
        apfel.SetAlphaEvolution("expanded")
    else:
        raise RuntimeError("ERROR: Unrecognised MODEV:", theory.get("ModEv"))

    # Coupling
    apfel.SetAlphaQCDRef(theory.get("alphas"), theory.get("Qref"))
    if theory.get("QED"):
        apfel.SetAlphaQEDRef(theory.get("alphaqed"), theory.get("Qedref"))

    # EW
    apfel.SetWMass(theory.get("MW"))
    apfel.SetZMass(theory.get("MZ"))
    apfel.SetGFermi(theory["GF"])
    apfel.SetSin2ThetaW(theory["SIN2TW"])

    apfel.SetCKM(*[float(x) for x in theory.get("CKM").split()])

    # TMCs
    apfel.SetProtonMass(theory.get("MP"))
    if theory.get("TMC"):
        apfel.EnableTargetMassCorrections(True)

    # Heavy Quark Masses
    if theory.get("HQ") == "POLE":
        apfel.SetPoleMasses(theory.get("mc"), theory.get("mb"), theory.get("mt"))
    elif theory.get("HQ") == "MSBAR":
        apfel.SetMSbarMasses(theory.get("mc"), theory.get("mb"), theory.get("mt"))
        apfel.SetMassScaleReference(
            theory.get("Qmc"), theory.get("Qmb"), theory.get("Qmt")
        )
    else:
        raise RuntimeError("Error: Unrecognised HQMASS")

    # Heavy Quark schemes
    apfel.SetMassScheme(theory.get("FNS"))
    apfel.EnableDampingFONLL(theory.get("DAMP"))
    if theory.get("FNS") == "FFNS":
        apfel.SetFFNS(theory.get("NfFF"))
        apfel.SetMassScheme("FFNS%d" % theory.get("NfFF"))
    else:
        apfel.SetVFNS()

    apfel.SetMaxFlavourAlpha(theory.get("MaxNfAs"))
    apfel.SetMaxFlavourPDFs(theory.get("MaxNfPdf"))

    # Scale ratios
    apfel.SetRenFacRatio(theory.get("XIR") / theory.get("XIF"))
    apfel.SetRenQRatio(theory.get("XIR"))
    apfel.SetFacQRatio(theory.get("XIF"))
    # Scale Variations
    # consistent with Evolution (0) or DIS only (1)
    # look at SetScaleVariationProcedure.f
    apfel.SetScaleVariationProcedure(theory.get("EScaleVar"))

    # Small-x resummation
    apfel.SetSmallxResummation(theory.get("SxRes"), theory.get("SxOrd"))
    apfel.SetMassMatchingScales(
        theory.get("kcThr"), theory.get("kbThr"), theory.get("ktThr")
    )

    # Intrinsic charm
    apfel.EnableIntrinsicCharm(theory.get("IC"))

    # Not included in the map
    #
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

    # set APFEL grid to ours
    if platform.node() in ["FHe19b", "topolinia-arch"]:
        apfel.SetNumberOfGrids(1)
        # create a 'double *' using swig wrapper
        yad_xgrid = operators["interpolation_xgrid"]
        xgrid = apfel.new_doubles(len(yad_xgrid))

        # fill the xgrid with
        for j, x in enumerate(yad_xgrid):
            apfel.doubles_setitem(xgrid, j, x)

        yad_deg = operators["interpolation_polynomial_degree"]
        # 1 = gridnumber
        apfel.SetExternalGrid(1, len(yad_xgrid) - 1, yad_deg, xgrid)

    # set DIS params
    apfel.SetPDFSet(pdf)
    # set Target

    # init evolution
    apfel.InitializeAPFEL()

    return apfel


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

    target_xgrid = operators["interpolation_xgrid"]
    pdf_name = pdf.set().name

    # Load apfel
    apf_start = time.perf_counter()
    apfel = load_apfel(theory, operators, pdf_name)
    print("Loading APFEL took %f s" % (time.perf_counter() - apf_start))

    # Run
    apf_tabs = {}
    for q2 in operators["Q2grid"]:

        apfel.EvolveAPFEL(theory["Q0"], np.sqrt(q2))
        print("Executing APFEL took %f s" % (time.perf_counter() - apf_start))

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
                    tab[pid] if pid in apf_tabs else np.zeros(len(target_xgrid))
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
