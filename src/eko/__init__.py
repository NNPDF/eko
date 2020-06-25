# -*- coding: utf-8 -*-

from eko._parameters import t_float, t_complex
import eko.runner


def run_dglap(setup):
    r"""
        This function takes a DGLAP theory configuration dictionary
        and performs the solution of the DGLAP equations.

        The EKO :math:`\hat E_{k,j}(t_1\leftarrow t_0)` is determined in order
        to fullfill the following evolution

        .. math::
            f(x_k,t_1) = \hat E_{k,j}(t_1\leftarrow t_0) f(x_j,t_0)

        Parameters
        ----------
            setup : dict
                input card - see :doc:`/Code/IO`

        Returns
        -------
            output : dict
                output dictionary - see :doc:`/Code/IO`
    """
    r = eko.runner.Runner(setup)
    output = r.get_output()
    return output
