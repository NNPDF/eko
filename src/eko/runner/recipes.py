"""Recipes containing instructions for atomic computations."""

from typing import List

from ..io.items import Evolution, Matching, Recipe
from ..io.struct import EKO
from ..io.types import EvolutionPoint as EPoint
from ..matchings import Atlas, Segment
from . import commons


def _elements(ep: EPoint, atlas: Atlas) -> List[Recipe]:
    """Determine recipes to compute a given operator."""
    recipes = []

    #  expanded = eko.operator_card.configs.scvar_method is sv.Modes.expanded
    #  mu2f = mu2 * eko.theory_card.xif**2 if expanded else mu2

    blocks = atlas.matched_path(ep)
    for block in blocks:
        if isinstance(block, Segment):
            cliff = block.target in atlas.walls
            recipe = Evolution.from_atlas(block, cliff=cliff)
        else:
            recipe = Matching.from_atlas(block)

        recipes.append(recipe)

    return recipes


def _create(evolgrid: List[EPoint], atlas: Atlas) -> List[Recipe]:
    """Create all associated recipes."""
    recipes = []
    for ep in evolgrid:
        recipes.extend(_elements(ep, atlas))

    return list(set(recipes))


def create(eko: EKO):
    """Create and load all associated recipes."""
    atlas = commons.atlas(eko.theory_card, eko.operator_card)
    recs = _create(eko.operator_card.evolgrid, atlas)
    eko.load_recipes(recs)
