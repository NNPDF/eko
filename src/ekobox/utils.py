"""Generic utilities to work with EKOs."""
import os
from typing import Optional

import numpy as np

from eko.io.struct import EKO, Operator

CONTRACTION = "ajbk,bkcl -> ajcl"


def ekos_product(
    eko_ini: EKO,
    eko_fin: EKO,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    path: Optional[os.PathLike] = None,
):
    """Compute the product of two ekos.

    Parameters
    ----------
    eko_ini :
        initial eko operator
    eko_fin :
        final eko operator
    rtol :
        relative tolerance on Q2, used to check compatibility
    atol :
        absolute tolerance on Q2, used to check compatibility
    path :
        if not provided, the operation is done in-place, otherwie a new
        operator is written at the given path

    """
    # TODO: add a control on the theory (but before we need to implement
    # another kind of output which includes the theory and operator runcards)

    q2match = eko_ini.approx(eko_fin.operator_card.mu0**2, rtol=rtol, atol=atol)
    if q2match is None:
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in"
            " the initial eko operator"
        )
    ope1 = eko_ini[q2match].operator.copy()
    ope1_error = eko_ini[q2match].error
    if ope1_error is not None:
        ope1_error = ope1_error.copy()

    if path is None:
        final_eko = eko_ini
    else:
        eko_ini.deepcopy(path)
        final_eko = EKO.edit(path)

    for q2, op2 in eko_fin.items():
        if q2 in eko_ini:
            continue

        op = np.einsum(CONTRACTION, ope1, op2.operator)

        if ope1_error is not None and op2.error is not None:
            error = np.einsum(CONTRACTION, ope1, op2.error) + np.einsum(
                CONTRACTION, ope1_error, op2.operator
            )
        else:
            error = None

        final_eko[q2] = Operator(operator=op, error=error)

    if path is not None:
        final_eko.close()
