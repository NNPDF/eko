# -*- coding: utf-8 -*-
import copy

import numpy as np

from eko.output import EKO, Operator


# TODO: add a control on the theory (but before we need to implement another
# kind of output which includes the theory and operator runcards)
def ekos_product(
    eko_ini: EKO, eko_fin: EKO, rtol: float = 1e-6, atol: float = 1e-10, in_place=True
) -> EKO:
    """Returns the product of two ekos

    Parameters
    ----------
    eko_ini : eko.output.Output
        initial eko operator
    eko_fin : eko.output.Output
        final eko operator
    rtol : float
        relative tolerance on Q2, used to check compatibility
    atol : float
        absolute tolerance on Q2, used to check compatibility
    in_place : bool
        do operation in place, modifying input arrays

    Returns
    -------
    eko.output.Output
        eko operator

    """
    q2match = eko_ini.approx(eko_fin.Q02, rtol=rtol, atol=atol)
    if q2match is None:
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in"
            " the initial eko operator"
        )
    ope1 = eko_ini[q2match].operator.copy()
    ope1_error = eko_ini[q2match].error.copy()

    ope2_dict = {}
    ope2_error_dict = {}
    for q2, op in eko_fin.items():
        ope2_dict[q2] = op.operator
        ope2_error_dict[q2] = op.error

    final_op_dict = {}
    final_op_error_dict = {}
    final_dict = {}
    for q2, op2 in ope2_dict.items():
        final_op_dict[q2] = np.einsum("ajbk,bkcl -> ajcl", ope1, op2)

        final_op_error_dict[q2] = np.einsum(
            "ajbk,bkcl -> ajcl", ope1, ope2_error_dict[q2]
        ) + np.einsum("ajbk,bkcl -> ajcl", ope1_error, op2)

        final_dict[q2] = {
            "operator": final_op_dict[q2],
            "error": final_op_error_dict[q2],
        }

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

        final_eko[q2] = Operator(operator=op, error=error)

    return final_eko
