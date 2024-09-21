"""Manipulate output generate by EKO."""

import copy
import logging
import warnings
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation
from ..interpolation import XGrid
from .struct import Operator

logger = logging.getLogger(__name__)

TARGETGRID_ROTATION = "ij,ajbk->aibk"
INPUTGRID_ROTATION = "ajbk,kl->ajbl"
SIMGRID_ROTATION = "ij,ajbk,kl->aibl"
"""Simultaneous grid rotation contraction indices."""


def rotation(
    new: Optional[XGrid], old: XGrid, check: Callable, compute: Callable
) -> npt.NDArray:
    """Define grid rotation.

    This function returns the necessary rotation,
    if the checks for a non-trivial new grid are passed.

    However, the check and the computation are delegated respectively to the
    callables `check` and `compute`.
    """
    if new is None:
        return None

    if check(new, old):
        warnings.warn("The new grid is close to the current one")
        return None

    return compute(new, old)


def xgrid_check(new: Optional[XGrid], old: XGrid) -> bool:
    """Check validity of new xgrid."""
    return new is not None and len(new) == len(old) and np.allclose(new.raw, old.raw)


def xgrid_compute_rotation(
    new: XGrid, old: XGrid, interpdeg: int, swap: bool = False
) -> npt.NDArray:
    """Compute rotation from old to new xgrid.

    By default, the roation is computed for a target xgrid. Whether the function
    should be used for an input xgrid, the `swap` argument should be set to
    `True`, in order to compute it in the other direction (i.e. the transposed).
    """
    if swap:
        new, old = old, new
    b = interpolation.InterpolatorDispatcher(old, interpdeg, False)
    return b.get_interpolation(new.raw)


def xgrid_reshape(
    elem: Operator,
    xgrid: XGrid,
    interpdeg: int,
    targetgrid: Optional[XGrid] = None,
    inputgrid: Optional[XGrid] = None,
) -> Operator:
    """Reinterpolate the operator on output and/or input grid(s).

    Target corresponds to the output PDF.
    """
    # calling with no arguments is an error
    if targetgrid is None and inputgrid is None:
        raise ValueError("Nor inputgrid nor targetgrid was given")

    check = xgrid_check
    crot = xgrid_compute_rotation

    # construct matrices
    targetrot = rotation(
        targetgrid,
        xgrid,
        check,
        lambda new, old: crot(new, old, interpdeg),
    )
    inputrot = rotation(
        inputgrid,
        xgrid,
        check,
        lambda new, old: crot(new, old, interpdeg, swap=True),
    )
    # after the checks: if there is still nothing to do, skip
    if targetrot is None and inputrot is None:
        logger.debug("Nothing done.")
        return copy.deepcopy(elem)

    # build new grid
    operands = [elem.operator]
    operands_errs = [elem.error]

    if targetrot is not None and inputrot is None:
        contraction = TARGETGRID_ROTATION
    elif inputrot is not None and targetrot is None:
        contraction = INPUTGRID_ROTATION
    else:
        contraction = SIMGRID_ROTATION

    if targetrot is not None:
        operands.insert(0, targetrot)
        operands_errs.insert(0, targetrot)
    if inputrot is not None:
        operands.append(inputrot)
        operands_errs.append(inputrot)

    new_operator = np.einsum(contraction, *operands, optimize="optimal")
    if elem.error is not None:
        new_error = np.einsum(contraction, *operands_errs, optimize="optimal")
    else:
        new_error = None

    return Operator(operator=new_operator, error=new_error)


TARGETPIDS_ROTATION = "ca,ajbk->cjbk"
INPUTPIDS_ROTATION = "ajbk,bd->ajdk"
SIMPIDS_ROTATION = "ca,ajbk,bd->cjdk"
"""Simultaneous grid rotation contraction indices."""


def flavor_reshape(
    elem: Operator,
    targetpids: Optional[npt.NDArray] = None,
    inputpids: Optional[npt.NDArray] = None,
) -> Operator:
    """Change the operator to have in the output targetpids and/or in the input
    inputpids.

    Parameters
    ----------
    elem :
        the operator to be rotated
    targetpids :
        target rotation specified in the flavor basis
    inputpids :
        input rotation specified in the flavor basis
    """
    # calling with no arguments is an error
    if targetpids is None and inputpids is None:
        raise ValueError("Nor inputpids nor targetpids was given")
    # now check to the current status
    if targetpids is not None and np.allclose(
        targetpids, np.eye(elem.operator.shape[0])
    ):
        targetpids = None
        warnings.warn("The new targetpids is close to current basis")
    if inputpids is not None and np.allclose(inputpids, np.eye(elem.operator.shape[2])):
        inputpids = None
        warnings.warn("The new inputpids is close to current basis")
    # after the checks: if there is still nothing to do, skip
    if targetpids is None and inputpids is None:
        logger.debug("Nothing done.")
        return copy.deepcopy(elem)

    # flip input around
    inv_inputpids = np.zeros_like(inputpids)
    if inputpids is not None:
        inv_inputpids = np.linalg.inv(inputpids)

    # build new grid
    ops = elem.operator
    errs = elem.error
    if targetpids is not None and inputpids is None:
        ops = np.einsum(TARGETPIDS_ROTATION, targetpids, ops, optimize="optimal")
        errs = (
            np.einsum(TARGETPIDS_ROTATION, targetpids, errs, optimize="optimal")
            if errs is not None
            else None
        )
    elif inputpids is not None and targetpids is None:
        ops = np.einsum(INPUTPIDS_ROTATION, ops, inv_inputpids, optimize="optimal")
        errs = (
            np.einsum(INPUTPIDS_ROTATION, errs, inv_inputpids, optimize="optimal")
            if errs is not None
            else None
        )
    else:
        ops = np.einsum(
            SIMPIDS_ROTATION, targetpids, ops, inv_inputpids, optimize="optimal"
        )
        errs = (
            np.einsum(
                SIMPIDS_ROTATION,
                targetpids,
                errs,
                inv_inputpids,
                optimize="optimal",
            )
            if errs is not None
            else None
        )

    return Operator(operator=ops, error=errs)


def to_evol(elem: Operator, source: bool = True, target: bool = False) -> Operator:
    """Rotate the operator into evolution basis.

    Parameters
    ----------
    elem :
        the operator to be rotated
    source :
        rotate on the input tensor
    target :
        rotate on the output tensor
    """
    # rotate
    inputpids = br.rotate_flavor_to_evolution if source else None
    targetpids = br.rotate_flavor_to_evolution if target else None
    return flavor_reshape(elem, inputpids=inputpids, targetpids=targetpids)


def to_uni_evol(elem: Operator, source: bool = True, target: bool = False) -> Operator:
    """Rotate the operator into evolution basis.

    Parameters
    ----------
    elem :
        the operator to be rotated
    source :
        rotate on the input tensor
    target :
        rotate on the output tensor
    """
    # rotate
    inputpids = br.rotate_flavor_to_unified_evolution if source else None
    targetpids = br.rotate_flavor_to_unified_evolution if target else None
    return flavor_reshape(elem, inputpids=inputpids, targetpids=targetpids)
