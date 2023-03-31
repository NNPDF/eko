"""Recipes containing instructions for atomic computations."""
from abc import ABC
from dataclasses import dataclass

from .. import EKO
from .. import scale_variations as sv
from ..io import runcards
from ..io.dictlike import DictLike
from ..io.types import SquaredScale
from ..thresholds import ThresholdsAtlas


@dataclass
class Recipe(DictLike, ABC):
    """Base recipe structure."""

    name: str


@dataclass
class EvolutionRecipe(Recipe):
    """Recipe compute evolution with a fixed number of light flavors."""

    final: bool
    mu20: SquaredScale
    mu2: SquaredScale


@dataclass
class MatchingRecipe(Recipe):
    """Recipe to compute the matching between two different flavor number schemes."""

    mu2: SquaredScale


def create(eko: EKO):
    """Create all associated recipes."""
    _ = eko.theory_card.matching

    masses = runcards.masses(
        eko.theory_card, eko.operator_card.configs.evolution_method
    )

    tc = ThresholdsAtlas(
        masses=masses,
        q2_ref=eko.operator_card.mu20,
        nf_ref=eko.theory_card.num_flavs_init,
        thresholds_ratios=None,
        max_nf=eko.theory_card.num_flavs_max_pdf,
    )

    for mu2 in eko.mu2grid:
        expanded = eko.operator_card.configs.scvar_method is sv.Modes.expanded
        mu2f = mu2 * eko.theory_card.xif**2 if expanded else mu2
