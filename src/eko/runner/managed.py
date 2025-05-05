"""Fully managed runner.

This is an automated runner, mainly suggested for small EKOs
computations.

The primitives used here to compute the various pieces are part of the
public interface, and should be directly used to manage a more complex
run for a considerably large operator.

Thus, parallelization and multi-node execution is possible using EKO
primitives, but not automatically performed.
"""

from pathlib import Path

from ..io.items import Target
from ..io.runcards import OperatorCard, TheoryCard
from ..io.struct import EKO
from . import operators, parts, recipes


def solve(theory: TheoryCard, operator: OperatorCard, path: Path):
    """Solve DGLAP equations in terms of evolution kernel operators (EKO)."""
    with EKO.create(path) as builder:
        eko = builder.load_cards(theory, operator).build()  # pylint: disable=E1101
        recipes.create(eko)

        for recipe in eko.recipes:
            eko.parts[recipe] = parts.evolve(eko, recipe)
            # flush the memory
            del eko.parts[recipe]
        for recipe in eko.recipes_matching:
            eko.parts_matching[recipe] = parts.match(eko, recipe)
            # flush the memory
            del eko.parts_matching[recipe]

        for ep in operator.evolgrid:
            components = operators.retrieve(ep, eko)
            target = Target.from_ep(ep)
            eko.operators[target] = operators.join(components)
            # flush the memory
            del eko.parts
            del eko.operators[target]
