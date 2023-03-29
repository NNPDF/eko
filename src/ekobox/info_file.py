"""LHAPDF info file utilities."""
import copy
import math

import numpy as np

from eko import couplings
from eko.io.runcards import OperatorCard, TheoryCard

from .genpdf import load


def build(
    theory_card: TheoryCard,
    operators_card: OperatorCard,
    num_members: int,
    info_update: dict,
):
    """Generate a lhapdf info file from theory and operators card.

    Parameters
    ----------
    theory_card : dict
        theory card
    operators_card : dict
        operators_card
    num_members : int
        number of pdf set members
    info_update : dict
        info to update

    Returns
    -------
    dict
        info file in lhapdf format
    """
    template_info = copy.deepcopy(load.template_info)
    template_info["SetDesc"] = "Evolved PDF from " + str(operators_card.mu0) + " GeV"
    template_info["Authors"] = ""
    template_info["FlavorScheme"] = "variable"
    template_info.update(info_update)
    template_info["NumFlavors"] = 14
    template_info["Flavors"] = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 21, 22]
    # TODO actually point to input grid
    template_info["XMin"] = float(operators_card.rotations.xgrid.raw[0])
    template_info["XMax"] = float(operators_card.rotations.xgrid.raw[-1])
    template_info["NumMembers"] = num_members
    template_info["OrderQCD"] = theory_card.order[0] - 1
    template_info["QMin"] = round(math.sqrt(operators_card.mu2grid[0]), 4)
    template_info["QMax"] = round(math.sqrt(operators_card.mu2grid[-1]), 4)
    template_info["MZ"] = theory_card.couplings.alphas.scale
    template_info["MUp"] = 0.0
    template_info["MDown"] = 0.0
    template_info["MStrange"] = 0.0
    template_info["MCharm"] = theory_card.quark_masses.c.value
    template_info["MBottom"] = theory_card.quark_masses.b.value
    template_info["MTop"] = theory_card.quark_masses.t.value
    template_info["AlphaS_MZ"] = theory_card.couplings.alphas.value
    template_info["AlphaS_OrderQCD"] = theory_card.order[0] - 1
    evmod = couplings.couplings_mod_ev(operators_card.configs.evolution_method)
    quark_masses = [(x.value) ** 2 for x in theory_card.quark_masses]
    sc = couplings.Couplings(
        theory_card.couplings,
        theory_card.order,
        evmod,
        quark_masses,
        hqm_scheme=theory_card.quark_masses_scheme,
        thresholds_ratios=np.power(list(iter(theory_card.matching)), 2.0),
    )
    alphas_values = np.array(
        [
            4.0
            * np.pi
            * sc.a_s(
                muf2,
            )
            for muf2 in operators_card.mu2grid
        ],
        dtype=float,
    )
    template_info["AlphaS_Vals"] = alphas_values.tolist()
    template_info["AlphaS_Qs"] = np.array(operators_card.mu2grid).tolist()
    return template_info
