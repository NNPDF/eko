# -*- coding: utf-8 -*-
"""EKO main module.

Please refer to our documentation for a full overview of the possibilities.
"""
from . import output, runner, version

# export public constants
__version__ = version.__version__

# export public methods
open = output.struct.EKO.open  # pylint: disable=redefined-builtin
create = output.struct.EKO.create


def run_dglap(theory_card: dict, operators_card: dict) -> output.EKO:
    r"""Solve DGLAP equations in terms of evolution kernel operators (EKO).

    The configuration as to be given as two configuration dictionaries.

    The EKO :math:`\mathbf E_{k,j}(a_s^1\leftarrow a_s^0)` is determined in order
    to fullfill the following evolution

    .. math::
        \mathbf f(x_k,a_s^1) = \mathbf E_{k,j}(a_s^1\leftarrow a_s^0) \mathbf f(x_j,a_s^0)

    Parameters
    ----------
    setup : dict
        input card - see :doc:`/code/IO`

    Returns
    -------
    output.EKO
        output object - see :doc:`/code/IO`
    """
    return runner.Runner(theory_card, operators_card).get_output()
