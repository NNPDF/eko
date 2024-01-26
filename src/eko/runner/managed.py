"""Fully managed runner.

This is an automated runner, mainly suggested for small EKOs computations.

The primitives used here to compute the various pieces are part of the public
interface, and should be directly used to manage a more complex run for a
considebaly large operator.

Thus, parallelization and multi-node execution is possible using EKO primitives,
but not automatically performed.

"""
from pathlib import Path

from ..io.items import Evolution, Matching, Target
from ..io.runcards import OperatorCard, TheoryCard
from ..io.struct import EKO
from . import commons, operators, parts, recipes


def solve(theory: TheoryCard, operator: OperatorCard, path: Path):
    """Solve DGLAP equations in terms of evolution kernel operators (EKO)."""
    theory.heavy.intrinsic_flavors = [4, 5, 6]

    with EKO.create(path) as builder:
        eko = builder.load_cards(theory, operator).build()  # pylint: disable=E1101

        atlas = commons.atlas(eko.theory_card, eko.operator_card)

        recs = recipes.create(eko.operator_card.evolgrid, atlas)
        eko.load_recipes(recs)

        for recipe in eko.recipes:
            assert isinstance(recipe, Evolution)
            eko.parts[recipe] = parts.evolve(eko, recipe)
            # flush the memory
            del eko.parts[recipe]
        for recipe in eko.recipes_matching:
            assert isinstance(recipe, Matching)
            eko.parts_matching[recipe] = parts.match(eko, recipe)
            # flush the memory
            del eko.parts_matching[recipe]

        for ep in operator.evolgrid:
            headers = recipes.elements(ep, atlas)
            parts_ = operators.retrieve(headers, eko.parts, eko.parts_matching)
            target = Target.from_ep(ep)
            eko.operators[target] = operators.join(parts_)
            # flush the memory
            del eko.parts
            del eko.parts_matching
            del eko.operators[target]

def solve_scet(theory: TheoryCard, operator: OperatorCard, path: Path):
    """Compute SCET matching kernels in terms of evolution kernel operators (EKO)."""
    with EKO.create(path) as builder:
        eko = builder.load_cards(theory, operator).build()  # pylint: disable=E1101
        
        # Only required info is the order in alpha_s and in log
        # This should be passed in the runcard
        # orders to be computed
        orders_alpha_L=[(1,0), (1,1), (1,2)]
        # number of actiave flavors(?) Required starting from nnlo
        nf = 5
        
        # create a recipe for the scet matching kernel and load it.
        rec = recipes.create_scet_recipe(orders_alpha_L)
        eko.load_recipes(rec)
        dummy_scale = 1000.
        # compute scet kernel
        for recipe in eko.recipes_scet:
            #eko.scet_kernels[recipe] = parts.scetI(eko, recipe)
            target = Target(dummy_scale, nf)
            eko.operators[target] = parts.scetI(eko, recipe)   # in order to load these object as ekos you need to save them in eko.operators using Target as indices
            dummy_scale += 1 
            #del eko.scet_kernels[recipe]
            del eko.operators[target]

        



