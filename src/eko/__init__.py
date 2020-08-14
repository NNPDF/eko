# -*- coding: utf-8 -*-

import eko.runner


def run_dglap(setup):
    r"""
        This function takes a DGLAP theory configuration dictionary
        and performs the solution of the DGLAP equations.

        The EKO :math:`\hat E_{k,j}(a_s^1\leftarrow a_s^0)` is determined in order
        to fullfill the following evolution

        .. math::
            f(x_k,a_s^1) = \hat E_{k,j}(a_s^1\leftarrow a_s^0) f(x_j,a_s^0)

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
