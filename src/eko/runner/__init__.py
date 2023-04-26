"""Manage steps to DGLAP solution, and operator creation."""
import os
from pathlib import Path
from typing import Union

from ..io import runcards
from ..io.runcards import OperatorCard, TheoryCard
from ..io.types import RawCard
from . import legacy, managed


# TODO: drop this altogether, replacing just with managed.solve
# it is currently kept not to break the interface, but the runcards upgrade and
# path conversion should be done by the caller, here we just clearly declare
# which types we expect
def solve(
    theory_card: Union[RawCard, TheoryCard],
    operators_card: Union[RawCard, OperatorCard],
    path: os.PathLike,
):
    r"""Solve DGLAP equations in terms of evolution kernel operators (EKO).

    The EKO :math:`\mathbf E_{k,j}(a_s^1\leftarrow a_s^0)` is determined in order
    to fullfill the following evolution

    .. math::
        \mathbf f(x_k,a_s^1) = \mathbf E_{k,j}(a_s^1\leftarrow a_s^0) \mathbf f(x_j,a_s^0)

    The configuration is split between the theory settings, representing
    Standard Model parameters and other defining features of the theory
    calculation, and the operator settings, those that are more closely related
    to the solution of the |DGLAP| equation itself, and determine the resulting
    operator features.

    Note
    ----
    For further information about EKO inputs and output see :doc:`/code/IO`

    """
    # TODO: drop this
    legacy.Runner(theory_card, operators_card, path).compute()


def solve_jets(
    theory_card: Union[RawCard, TheoryCard],
    operators_card: Union[RawCard, OperatorCard],
    path: os.PathLike,
):
    """Implement solve for new output."""
    new_theory, new_operator = runcards.update(theory_card, operators_card)
    managed.solve(new_theory, new_operator, Path(path))


#  solve = solve_jets
