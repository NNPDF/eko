"""Recipes containing instructions for atomic computations."""

from typing import List

from ..io.items import Evolution, Matching, Recipe
from ..io.types import EvolutionPoint as EPoint
from ..matchings import Atlas, Segment


def elements(ep: EPoint, atlas: Atlas) -> List[Recipe]:
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


def create(evolgrid: List[EPoint], atlas: Atlas) -> List[Recipe]:
    """Create all associated recipes."""
    recipes = []
    for ep in evolgrid:
        recipes.extend(elements(ep, atlas))

    return list(set(recipes))
