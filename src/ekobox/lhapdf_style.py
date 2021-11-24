import io
import pathlib

import banana
import yaml


def create_info_file(theory_card, operators_card, NumMembers, info_update):
    """
    Generate a lhapdf info file from theory and operators card

    Parameters
    ----------
        theory_card : dict
            theory card
        operators_card : dict
            operators_card
        NumMembers : int
            number of pdf set members
        info_update : dict
            info to update

    Returns
    -------
        : dict
        info file in lhapdf format
    """
    here = pathlib.Path(banana.__file__).parent / "data/genpdf"
    with open(here / "templatePDF.info", "r") as o:
        template_info = yaml.safe_load(o)
    template_info["SetDesc"] = "Evolved PDF from " + str(theory_card["Q0"]) + " GeV"
    template_info["Authors"] = ""

    template_info.update(info_update)
    template_info["NumFlavors"] = theory_card["MaxNfPdf"]
    template_info["XMin"] = operators_card["interpolation_xgrid"][0]
    template_info["XMax"] = operators_card["interpolation_xgrid"][-1]
    # remember Flavors and FlavorScheme
    template_info["NumMembers"] = NumMembers
    template_info["OrderQCD"] = theory_card["PTO"]
    template_info["QMin"] = operators_card["Q2grid"][0]
    template_info["QMax"] = operators_card["Q2grid"][-1]
    template_info["MZ"] = theory_card["MZ"]
    template_info["MUp"] = 0.0
    template_info["MDown"] = 0.0
    template_info["MStrange"] = 0.0
    template_info["MCharm"] = theory_card["mc"]
    template_info["MBottom"] = theory_card["mb"]
    template_info["MTop"] = theory_card["mt"]
    template_info["AlphaS_MZ"] = theory_card["alphas"]
    template_info["AlphaS_OrderQCD"] = theory_card["PTO"]
    # NB: Maybe I need the actual operator to obtain AlphaS_Qs
    return template_info
