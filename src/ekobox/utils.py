"""Generic utilities to work with EKOs."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from eko.io.struct import EKO, Operator

CONTRACTION = "ajbk,bkcl -> ajcl"


def ekos_product(
    eko_ini: EKO,
    eko_fin: EKO,
    rtol: float = 1e-6,
    atol: float = 1e-10,
    path: Optional[Path] = None,
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

    ep_match = eko_ini.approx(
        (eko_fin.operator_card.init[0] ** 2, eko_fin.operator_card.init[1]),
        rtol=rtol,
        atol=atol,
    )
    if ep_match is None:
        raise ValueError(
            "Initial Q2 of final eko operator does not match any final Q2 in"
            " the initial eko operator"
        )
    ope1 = eko_ini[ep_match]
    assert ope1 is not None

    ope1_value = ope1.operator.copy()
    ope1_error = ope1.error
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

        op = np.einsum(CONTRACTION, ope1_value, op2.operator)

        if ope1_error is not None and op2.error is not None:
            error = np.einsum(CONTRACTION, ope1_value, op2.error) + np.einsum(
                CONTRACTION, ope1_error, op2.operator
            )
        else:
            error = None

        final_eko[q2] = Operator(operator=op, error=error)

    if path is not None:
        final_eko.close()


def regroup_evolgrid(evolgrid: list):
    """Split evolution points by nf and sort by scale."""
    by_nf = defaultdict(list)
    for q, nf in sorted(evolgrid, key=lambda ep: ep[1]):
        by_nf[nf].append(q)
    return {nf: sorted(qs) for nf, qs in by_nf.items()}
