"""Fully managed runner.

This is an automated runner, mainly suggested for small EKOs computations.

The primitives used here to compute the various pieces are part of the public
interface, and should be directly used to manage a more complex run for a
considebaly large operator.

Thus, parallelization and multi-node execution is possible using EKO primitives,
but not automatically performed.

"""

from pathlib import Path

from ..io.runcards import OperatorCard, TheoryCard
from ..io.struct import EKO
from . import recipes


def solve(theory: TheoryCard, operator: OperatorCard, path: Path):
    """Solve DGLAP equations in terms of evolution kernel operators (EKO)."""
    with EKO.create(path) as builder:
        eko = builder.load_cards(theory, operator).build()

        recipes.create(eko)

        #  for recipe in eko.recipes:
        #  pass
