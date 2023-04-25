"""Manipulate output generate by EKO."""
import logging
import warnings
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from .. import basis_rotation as br
from .. import interpolation
from ..interpolation import XGrid
from .struct import EKO

logger = logging.getLogger(__name__)

TARGETGRID_ROTATION = "ij,ajbk->aibk"
INPUTGRID_ROTATION = "ajbk,kl->ajbl"
SIMGRID_ROTATION = "ij,ajbk,kl->aibl"
"""Simultaneous grid rotation contraction indices."""

Basis = Union[XGrid, npt.NDArray]


def rotation(new: Optional[Basis], old: Basis, check: Callable, compute: Callable):
    """Define grid rotation.

    This function returns the new grid to be assigned and the rotation computed,
    if the checks for a non-trivial new grid are passed.

    However, the check and the computation are delegated respectively to the
    callables `check` and `compute`.

    """
    if new is None:
        return old, None

    if check(new, old):
        warnings.warn("The new grid is close to the current one")
        return old, None

    return new, compute(new, old)


def xgrid_check(new: Optional[XGrid], old: XGrid):
    """Check validity of new xgrid."""
    return new is not None and len(new) == len(old) and np.allclose(new.raw, old.raw)


def xgrid_compute_rotation(new: XGrid, old: XGrid, interpdeg: int, swap: bool = False):
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
    eko: EKO,
    targetgrid: Optional[XGrid] = None,
    inputgrid: Optional[XGrid] = None,
):
    """Reinterpolate operators on output and/or input grids.

    Target corresponds to the output PDF.

    The operation is in-place.

    """
    eko.assert_permissions(write=True)

    # calling with no arguments is an error
    if targetgrid is None and inputgrid is None:
        raise ValueError("Nor inputgrid nor targetgrid was given")

    interpdeg = eko.operator_card.configs.interpolation_polynomial_degree
    check = xgrid_check
    crot = xgrid_compute_rotation

    # construct matrices
    newtarget, targetrot = rotation(
        targetgrid,
        eko.rotations.targetgrid,
        check,
        lambda new, old: crot(new, old, interpdeg),
    )
    newinput, inputrot = rotation(
        inputgrid,
        eko.rotations.inputgrid,
        check,
        lambda new, old: crot(new, old, interpdeg, swap=True),
    )

    # after the checks: if there is still nothing to do, skip
    if targetrot is None and inputrot is None:
        logger.debug("Nothing done.")
        return
    # if no rotation is done, the grids are not modified
    if targetrot is not None:
        eko.rotations.targetgrid = newtarget
    if targetrot is not None:
        eko.rotations.targetgrid = newinput

    # build new grid
    for ep, elem in eko.items():
        assert elem is not None

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

        elem.operator = np.einsum(contraction, *operands, optimize="optimal")
        if elem.error is not None:
            elem.error = np.einsum(contraction, *operands_errs, optimize="optimal")

        eko[ep] = elem

    eko.update()


TARGETPIDS_ROTATION = "ca,ajbk->cjbk"
INPUTPIDS_ROTATION = "ajbk,bd->ajdk"
SIMPIDS_ROTATION = "ca,ajbk,bd->cjdk"
"""Simultaneous grid rotation contraction indices."""


def flavor_reshape(
    eko: EKO,
    targetpids: Optional[npt.NDArray] = None,
    inputpids: Optional[npt.NDArray] = None,
    update: bool = True,
):
    """Change the operators to have in the output targetpids and/or in the input inputpids.

    The operation is in-place.

    Parameters
    ----------
    eko :
        the operator to be rotated
    targetpids :
        target rotation specified in the flavor basis
    inputpids :
        input rotation specified in the flavor basis
    update :
        update :class:`~eko.io.struct.EKO` metadata after writing

    """
    eko.assert_permissions(write=True)

    # calling with no arguments is an error
    if targetpids is None and inputpids is None:
        raise ValueError("Nor inputpids nor targetpids was given")
    # now check to the current status
    if targetpids is not None and np.allclose(
        targetpids, np.eye(len(eko.rotations.targetpids))
    ):
        targetpids = None
        warnings.warn("The new targetpids is close to current basis")
    if inputpids is not None and np.allclose(
        inputpids, np.eye(len(eko.rotations.inputpids))
    ):
        inputpids = None
        warnings.warn("The new inputpids is close to current basis")
    # after the checks: if there is still nothing to do, skip
    if targetpids is None and inputpids is None:
        logger.debug("Nothing done.")
        return

    # flip input around
    if inputpids is not None:
        inv_inputpids = np.linalg.inv(inputpids)

    # build new grid
    for q2, elem in eko.items():
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
        elem.operator = ops
        elem.error = errs

        eko[q2] = elem

    # drop PIDs - keeping them int nevertheless
    # there is no meaningful way to set them in general, after rotation
    if inputpids is not None:
        eko.rotations.inputpids = np.array([0] * len(eko.rotations.inputpids))
    if targetpids is not None:
        eko.rotations.targetpids = np.array([0] * len(eko.rotations.targetpids))

    if update:
        eko.update()


def to_evol(eko: EKO, source: bool = True, target: bool = False):
    """Rotate the operator into evolution basis.

    This also assigns also the pids. The operation is in-place.

    Parameters
    ----------
    eko :
        the operator to be rotated
    source :
        rotate on the input tensor
    target :
        rotate on the output tensor

    """
    # rotate
    inputpids = br.rotate_flavor_to_evolution if source else None
    targetpids = br.rotate_flavor_to_evolution if target else None
    # prevent metadata update, since flavor_reshape has not enough information
    # to determine inpupids and targetpids, and they will be updated after the
    # call
    flavor_reshape(eko, inputpids=inputpids, targetpids=targetpids, update=False)
    # assign pids
    if source:
        eko.rotations.inputpids = inputpids
    if target:
        eko.rotations.targetpids = targetpids

    eko.update()


def to_uni_evol(eko: EKO, source: bool = True, target: bool = False):
    """Rotate the operator into evolution basis.

    This also assigns also the pids. The operation is in-place.

    Parameters
    ----------
    eko :
        the operator to be rotated
    source :
        rotate on the input tensor
    target :
        rotate on the output tensor

    """
    # rotate
    inputpids = br.rotate_flavor_to_unified_evolution if source else None
    targetpids = br.rotate_flavor_to_unified_evolution if target else None
    # prevent metadata update, since flavor_reshape has not enough information
    # to determine inpupids and targetpids, and they will be updated after the
    # call
    flavor_reshape(eko, inputpids=inputpids, targetpids=targetpids, update=False)
    # assign pids
    if source:
        eko.rotations.inputpids = inputpids
    if target:
        eko.rotations.targetpids = targetpids

    eko.update()
