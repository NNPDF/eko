"""Runners common utilities."""

import numpy as np

from ..couplings import Couplings, couplings_mod_ev
from ..interpolation import InterpolatorDispatcher
from ..io import runcards
from ..io.runcards import OperatorCard, TheoryCard
from ..io.types import ScaleVariationsMethod
from ..matchings import Atlas

BANNER = r"""
oooooooooooo oooo    oooo  \\ .oooooo.
`888'     `8 `888   .8P'  //////    `Y8b
 888          888  d8'   \\o\/////    888
 888oooo8     88888     \\\\/8/////   888
 888    "     888`88b.      888 ///   888
 888       o  888  `88b.    `88b //  d88'
o888ooooood8 o888o  o888o     `Y8bood8P'
"""


def interpolator(operator: OperatorCard) -> InterpolatorDispatcher:
    """Create interpolator from runcards."""
    return InterpolatorDispatcher(
        xgrid=operator.xgrid,
        polynomial_degree=operator.configs.interpolation_polynomial_degree,
    )


def atlas(theory: TheoryCard, operator: OperatorCard) -> Atlas:
    """Create thresholds atlas from runcards."""
    # TODO: cache result
    masses = runcards.masses(theory, operator.configs.evolution_method)
    matching_scales = np.power(theory.heavy.matching_ratios, 2.0) * np.array(masses)
    return Atlas(matching_scales.tolist(), (operator.mu20, operator.init[1]))


def couplings(theory: TheoryCard, operator: OperatorCard) -> Couplings:
    """Create couplings from runcards."""
    thresholds_ratios = np.power(theory.heavy.matching_ratios, 2.0)
    masses = runcards.masses(theory, operator.configs.evolution_method)
    return Couplings(
        couplings=theory.couplings,
        order=theory.order,
        method=couplings_mod_ev(operator.configs.evolution_method),
        masses=masses,
        hqm_scheme=theory.heavy.masses_scheme,
        thresholds_ratios=thresholds_ratios
        * (
            theory.xif**2
            if operator.configs.scvar_method == ScaleVariationsMethod.EXPONENTIATED
            else 1.0
        ),
    )
