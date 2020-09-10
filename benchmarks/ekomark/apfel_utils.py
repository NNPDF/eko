import platform

import eko


def load_apfel(theory, pdf="ToyLH"):
    """
    Set APFEL parameter from ``theory`` dictionary.

    Parameters
    ----------
    theory : dict
        theory and process parameters

    Returns
    -------
    module
        loaded apfel wrapper
    """
    import apfel  # pylint:disable=import-outside-toplevel

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
        yad_xgrid = theory["interpolation_xgrid"]
        if yad_xgrid[0] == "make_grid":
            yad_xgrid = eko.interpolation.make_grid(*yad_xgrid[1:])
        xgrid = apfel.new_doubles(len(yad_xgrid))

        # fill the xgrid with
        for j, x in enumerate(yad_xgrid):
            apfel.doubles_setitem(xgrid, j, x)

        yad_deg = theory["interpolation_polynomial_degree"]
        # 1 = gridnumber
        apfel.SetExternalGrid(1, len(yad_xgrid) - 1, yad_deg, xgrid)

    # set DIS params
    apfel.SetPDFSet(pdf)
    # set Target

    # init evolution
    apfel.InitializeAPFEL()

    return apfel
