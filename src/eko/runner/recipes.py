"""Recipes containing instructions for atomic computations."""
from abc import ABC
from dataclasses import dataclass

from .. import EKO
from ..io.dictlike import DictLike
from ..io.types import SquaredScale


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
    matching = eko.theory_card.matching
