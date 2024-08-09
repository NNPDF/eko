"""LHAPDF info file utilities."""

import copy
import math
from typing import Any, Dict

import numpy as np

from eko import couplings
from eko.io.runcards import OperatorCard, TheoryCard

from .genpdf import load
from .utils import regroup_evolgrid


def build(
    theory_card: TheoryCard,
    operators_card: OperatorCard,
    num_members: int,
    info_update: dict,
) -> dict:
    """Generate a lhapdf info file.

    Parameters
    ----------
    theory_card :
        theory card
    operators_card :
        operators card
    num_members :
        number of pdf set members
    info_update :
        additional info to update

    Returns
    -------
    dict
        info file in lhapdf format
    """
    template_info = copy.deepcopy(load.template_info)
    template_info["SetDesc"] = (
        "Evolved PDF from " + str(operators_card.init[0]) + " GeV"
    )
    template_info["Authors"] = ""
    template_info["FlavorScheme"] = "variable"
    template_info.update(info_update)
    template_info["NumFlavors"] = max(nf for _, nf in operators_card.mugrid)
    template_info["Flavors"] = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 21, 22]
    # TODO actually point to input grid
    template_info["XMin"] = float(operators_card.xgrid.raw[0])
    template_info["XMax"] = float(operators_card.xgrid.raw[-1])
    template_info["NumMembers"] = num_members
    template_info["OrderQCD"] = theory_card.order[0] - 1
    template_info["QMin"] = round(math.sqrt(operators_card.mu2grid[0]), 4)
    template_info["QMax"] = round(math.sqrt(operators_card.mu2grid[-1]), 4)
    template_info["MZ"] = theory_card.couplings.ref[0]
    template_info["MUp"] = 0.0
    template_info["MDown"] = 0.0
    template_info["MStrange"] = 0.0
    template_info["MCharm"] = theory_card.heavy.masses.c.value
    template_info["MBottom"] = theory_card.heavy.masses.b.value
    template_info["MTop"] = theory_card.heavy.masses.t.value
    # dump alphas
    template_info.update(build_alphas(theory_card, operators_card))
    return template_info


def build_alphas(
    theory_card: TheoryCard,
    operators_card: OperatorCard,
) -> dict:
    """Generate a couplings section of lhapdf info file.

    Parameters
    ----------
    theory_card : dict
        theory card
    operators_card : dict
        operators card

    Returns
    -------
    dict
        info file section in lhapdf format
    """
    # start with meta stuff
    template_info: Dict[str, Any] = {}
    template_info["AlphaS_MZ"] = theory_card.couplings.alphas
    template_info["AlphaS_OrderQCD"] = theory_card.order[0] - 1
    # prepare
    evolgrid = regroup_evolgrid(operators_card.mugrid)
    evmod = couplings.couplings_mod_ev(operators_card.configs.evolution_method)
    quark_masses = [(x.value) ** 2 for x in theory_card.heavy.masses]
    sc = couplings.Couplings(
        theory_card.couplings,
        theory_card.order,
        evmod,
        quark_masses,
        hqm_scheme=theory_card.heavy.masses_scheme,
        thresholds_ratios=np.power(list(iter(theory_card.heavy.matching_ratios)), 2.0),
    )
    # add actual values
    alphas_values = []
    alphas_qs = []
    for nf, mus in evolgrid.items():
        for mu in mus:
            alphas_values.append(float(4.0 * np.pi * sc.a_s(mu * mu, nf_to=nf)))
            alphas_qs.append(mu)

    template_info["AlphaS_Vals"] = alphas_values
    template_info["AlphaS_Qs"] = alphas_qs
    return template_info
