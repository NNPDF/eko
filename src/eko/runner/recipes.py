"""Recipes containing instructions for atomic computations."""
from typing import List

from .. import EKO
from ..io.items import Evolution, Matching, Recipe
from ..matchings import Segment
from . import commons


def create(eko: EKO) -> List[Recipe]:
    """Create all associated recipes."""
    atlas = commons.atlas(eko.theory_card, eko.operator_card)

    recipes = []
    for ep in eko:
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
