"""Evolution Kernel Operators."""
from . import io, runner, version
from .io.struct import EKO

__version__ = version.__version__


def run_dglap(theory_card, operators_card):
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
    theory_card : dict
        theory parameters and related settings
    operator_card : dict
        solution configurations, and further EKO options

    Returns
    -------
    EKO
        computed operator

    Note
    ----
        For further information about EKO inputs and output see :doc:`/code/IO`

    """
    return runner.Runner(theory_card, operators_card).get_output()
