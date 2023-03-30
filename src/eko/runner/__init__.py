"""Manage steps to DGLAP solution, and operator creation."""
import os
from typing import Union

from ..io.runcards import OperatorCard, TheoryCard
from ..io.types import RawCard
from . import legacy


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

    Parameters
    ----------
    theory_card :
        theory parameters and related settings
    operator_card :
        solution configurations, and further EKO options
    path :
        path where to store the computed operator

    Note
    ----
    For further information about EKO inputs and output see :doc:`/code/IO`

    """
    legacy.Runner(theory_card, operators_card, path).compute()
