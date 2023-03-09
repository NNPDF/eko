"""Manipulate output generate by EKO."""
import logging
import warnings
from typing import Optional

import numpy as np

from .. import basis_rotation as br
from .. import interpolation
from .struct import EKO

logger = logging.getLogger(__name__)

TARGETGRID_ROTATION = "ij,ajbk->aibk"
INPUTGRID_ROTATION = "ajbk,kl->ajbl"
SIMGRID_ROTATION = "ij,ajbk,kl->aibl"
"""Simultaneous grid rotation contraction indices."""


def xgrid_reshape(
    eko: EKO,
    targetgrid: Optional[interpolation.XGrid] = None,
    inputgrid: Optional[interpolation.XGrid] = None,
):
    """Reinterpolate operators on output and/or input grids.

    The operation is in-place.

    Parameters
    ----------
    eko :
        the operator to be rotated
    targetgrid :
        xgrid for the target (output PDF)
    inputgrid :
        xgrid for the input (input PDF)

    """
    eko.assert_permissions(write=True)

    # calling with no arguments is an error
    if targetgrid is None and inputgrid is None:
        raise ValueError("Nor inputgrid nor targetgrid was given")
    # now check to the current status
    if (
        targetgrid is not None
        and len(targetgrid) == len(eko.rotations.targetgrid)
        and np.allclose(targetgrid.raw, eko.rotations.targetgrid.raw)
    ):
        targetgrid = None
        warnings.warn("The new targetgrid is close to the current targetgrid")
    if (
        inputgrid is not None
        and len(inputgrid) == len(eko.rotations.inputgrid)
        and np.allclose(inputgrid.raw, eko.rotations.inputgrid.raw)
    ):
        inputgrid = None
        warnings.warn("The new inputgrid is close to the current inputgrid")
    # after the checks: if there is still nothing to do, skip
    if targetgrid is None and inputgrid is None:
        logger.debug("Nothing done.")
        return

    # construct matrices
    if targetgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            eko.rotations.targetgrid,
            eko.operator_card.configs.interpolation_polynomial_degree,
            False,
        )
        target_rot = b.get_interpolation(targetgrid.raw)
        eko.rotations.targetgrid = targetgrid
    if inputgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            inputgrid,
            eko.operator_card.configs.interpolation_polynomial_degree,
            False,
        )
        input_rot = b.get_interpolation(eko.rotations.inputgrid.raw)
        eko.rotations.inputgrid = inputgrid

    # build new grid
    for q2, elem in eko.items():
        ops = elem.operator
        errs = elem.error
        if targetgrid is not None and inputgrid is None:
            ops = np.einsum(TARGETGRID_ROTATION, target_rot, ops, optimize="optimal")
            errs = (
                np.einsum(TARGETGRID_ROTATION, target_rot, errs, optimize="optimal")
                if errs is not None
                else None
            )
        elif inputgrid is not None and targetgrid is None:
            ops = np.einsum(INPUTGRID_ROTATION, ops, input_rot, optimize="optimal")
            errs = (
                np.einsum(INPUTGRID_ROTATION, errs, input_rot, optimize="optimal")
                if errs is not None
                else None
            )
        else:
            ops = np.einsum(
                SIMGRID_ROTATION, target_rot, ops, input_rot, optimize="optimal"
            )
            errs = (
                np.einsum(
                    SIMGRID_ROTATION, target_rot, errs, input_rot, optimize="optimal"
                )
                if errs is not None
                else None
            )
        elem.operator = ops
        elem.error = errs

        eko[q2] = elem

    eko.update()


TARGETPIDS_ROTATION = "ca,ajbk->cjbk"
INPUTPIDS_ROTATION = "ajbk,bd->ajdk"
SIMPIDS_ROTATION = "ca,ajbk,bd->cjdk"
"""Simultaneous grid rotation contraction indices."""


def flavor_reshape(
    eko: EKO,
    targetpids: Optional[np.ndarray] = None,
    inputpids: Optional[np.ndarray] = None,
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
