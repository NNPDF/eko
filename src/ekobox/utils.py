import numpy as np
import copy 

# TODO: add a control on the theory (but before we need to implement another kind of output which includes the theory and operator runcards)
def ekos_product(eko_ini, eko_fin, in_place=True, force=False):
    """
    Returns te product of two ekos

    Parameters
    ----------
        op_eko_ini : eko.output.Output
            initial eko operator
        op_eko_fin : eko.output.Output
            final eko operator

    Returns
    -------
        : eko.output.Output
            eko operator
    """
    if eko_fin["q2_ref"] not in eko_ini["Q2grid"].keys():
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in the initial eko operator"
        )
    ope1 = eko_ini["Q2grid"][eko_fin["q2_ref"]]["operators"]
    ope1_error = eko_ini["Q2grid"][eko_fin["q2_ref"]]["operator_errors"]
    ope2_dict = {}
    ope2_error_dict = {}
    for Q2 in eko_fin["Q2grid"].keys():
        ope2_dict[Q2] = eko_fin["Q2grid"][Q2]["operators"]
        ope2_error_dict[Q2] = eko_fin["Q2grid"][Q2]["operator_errors"]
    final_op_dict = {}
    final_op_error_dict = {}
    final_alphas_dict = {}
    final_dict = {}
    for Q2 in ope2_dict.keys():
        final_op_dict[Q2] = np.einsum("ajbk,bkcl -> ajcl", ope1, ope2_dict[Q2])
        final_op_error_dict[Q2] = np.add(np.einsum(
            "ajbk,bkcl -> ajcl", ope1, ope2_error_dict[Q2]
        ), np.einsum(
            "ajbk,bkcl -> ajcl", ope1_error, ope2_dict[Q2]))
        final_alphas_dict[Q2] = eko_fin["Q2grid"][Q2]["alphas"]
        final_dict[Q2] = {
            "operators": final_op_dict[Q2],
            "operator_errors": final_op_error_dict[Q2],
            "alphas": final_alphas_dict[Q2],
        }
    final_eko = None
    if in_place is False:
        final_eko = copy.deepcopy(eko_ini)
    else:
        final_eko = eko_ini
    final_eko["Q2grid"] = final_dict
    return final_eko
