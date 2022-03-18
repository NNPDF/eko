# -*- coding: utf-8 -*-
import copy

import numpy as np

from eko.output import EKO


# TODO: add a control on the theory (but before we need to implement another
# kind of output which includes the theory and operator runcards)
def ekos_product(eko_ini: EKO, eko_fin: EKO, in_place=True) -> EKO:
    """Returns the product of two ekos

    Parameters
    ----------
    eko_ini : eko.output.Output
        initial eko operator
    eko_fin : eko.output.Output
        final eko operator
    in_place : bool
        do operation in place, modifying input arrays

    Returns
    -------
    eko.output.Output
        eko operator

    """
    # TODO: check if it's close, instead of checking identity
    if eko_fin.Q02 not in eko_ini.Q2grid:
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in"
            " the initial eko operator"
        )
    ope1 = eko_ini[eko_fin.Q02].operator.copy()
    ope1_error = eko_ini[eko_fin.Q02].error.copy()

    if in_place is False:
        final_eko = copy.deepcopy(eko_ini)
    else:
        final_eko = eko_ini

    for q2, op2 in eko_fin.items():
        if q2 in eko_ini:
            continue

        op = np.einsum("ajbk,bkcl -> ajcl", ope1, op2.operator)

        error = np.einsum("ajbk,bkcl -> ajcl", ope1, op2.error) + np.einsum(
            "ajbk,bkcl -> ajcl", ope1_error, op2.operator
        )

        alphas = eko_fin[q2].alphas
        final_eko[q2] = dict(operator=op, error=error, alphas=alphas)

    return final_eko
