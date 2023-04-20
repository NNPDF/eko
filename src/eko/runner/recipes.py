"""Recipes containing instructions for atomic computations."""
from abc import ABC
from dataclasses import dataclass

from .. import EKO
from .. import scale_variations as sv
from ..io.dictlike import DictLike
from ..io.types import SquaredScale
from . import commons


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
    tc = commons.threshold_atlas(eko.theory_card, eko.operator_card)

    terminal = []
    for ep in eko:
        #  expanded = eko.operator_card.configs.scvar_method is sv.Modes.expanded
        #  mu2f = mu2 * eko.theory_card.xif**2 if expanded else mu2

        blocks = tc.path(*ep)
        terminal.append(blocks.pop())
