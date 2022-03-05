# -*- coding: utf-8 -*-
import copy

import numpy as np


# TODO: add a control on the theory (but before we need to implement another
# kind of output which includes the theory and operator runcards)
def ekos_product(eko_ini, eko_fin, in_place=True):
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
    if eko_fin["q2_ref"] not in eko_ini["Q2grid"].keys():
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in"
            " the initial eko operator"
        )
    ope1 = eko_ini["Q2grid"][eko_fin["q2_ref"]]["operators"]
    ope1_error = eko_ini["Q2grid"][eko_fin["q2_ref"]]["operator_errors"]

    ope2_dict = {}
    ope2_error_dict = {}
    for q2, op in eko_fin["Q2grid"].items():
        ope2_dict[q2] = op["operators"]
        ope2_error_dict[q2] = op["operator_errors"]

    final_op_dict = {}
    final_op_error_dict = {}
    final_alphas_dict = {}
    final_dict = {}
    for q2, op2 in ope2_dict.items():
        final_op_dict[q2] = np.einsum("ajbk,bkcl -> ajcl", ope1, op2)

        final_op_error_dict[q2] = np.einsum(
            "ajbk,bkcl -> ajcl", ope1, ope2_error_dict[q2]
        ) + np.einsum("ajbk,bkcl -> ajcl", ope1_error, op2)

        final_alphas_dict[q2] = eko_fin["Q2grid"][q2]["alphas"]
        final_dict[q2] = {
            "operators": final_op_dict[q2],
            "operator_errors": final_op_error_dict[q2],
            "alphas": final_alphas_dict[q2],
        }

    final_eko = None
    if in_place is False:
        final_eko = copy.deepcopy(eko_ini)
    else:
        final_eko = eko_ini
    final_eko["Q2grid"] = final_dict
    return final_eko
