"""Fully managed runner.

This is an automated runner, mainly suggested for small EKOs computations.

The primitives used here to compute the various pieces are part of the public
interface, and should be directly used to manage a more complex run for a
considebaly large operator.

Thus, parallelization and multi-node execution is possible using EKO primitives,
but not automatically performed.

"""
from itertools import chain
from pathlib import Path

from ..io.items import Target
from ..io.runcards import OperatorCard, TheoryCard
from ..io.struct import EKO
from . import commons, operators, parts, recipes


def solve(theory: TheoryCard, operator: OperatorCard, path: Path):
    """Solve DGLAP equations in terms of evolution kernel operators (EKO)."""
    with EKO.create(path) as builder:
        eko = builder.load_cards(theory, operator).build()

        recs = recipes.create(eko)
        eko.load_recipes(recs)

        for recipe in chain(eko.recipes, eko.recipes_matching):
            parts.compute(eko, recipe)

        atlas = commons.atlas(eko.theory_card, eko.operator_card)

        for ep in operator.evolgrid:
            headers = recipes.elements(ep, atlas)
            parts_ = operators.retrieve(eko, headers)
            eko.operators[Target.from_ep(ep)] = operators.join(eko, parts_)
