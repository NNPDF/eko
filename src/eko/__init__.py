"""Evolution Kernel Operators."""

from . import output, version
from .runner import legacy as runner

__version__ = version.__version__


def run_dglap(theory_card, operators_card):
    r"""Compute evolution operator from runcards.

    This function takes a DGLAP theory configuration dictionary
    and performs the solution of the DGLAP equations.

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
    output : dict
        output dictionary - see :doc:`/code/IO`

    """
    r = runner.Runner(theory_card, operators_card)
    out = r.get_output()
    return out
