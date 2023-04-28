"""Recipes containing instructions for atomic computations."""
from .. import EKO
from . import commons


def create(eko: EKO):
    """Create all associated recipes."""
    atlas = commons.atlas(eko.theory_card, eko.operator_card)

    terminal = []
    for ep in eko:
        #  expanded = eko.operator_card.configs.scvar_method is sv.Modes.expanded
        #  mu2f = mu2 * eko.theory_card.xif**2 if expanded else mu2

        blocks = atlas.path(ep)
        terminal.append(blocks.pop())
