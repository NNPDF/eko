"""Manipulate output generate by EKO."""
import logging
import warnings
from typing import Optional

import numpy as np

from .. import basis_rotation as br
from .. import interpolation
from .struct import EKO

logger = logging.getLogger(__name__)


def xgrid_reshape(
    eko: EKO,
    targetgrid: Optional[interpolation.XGrid] = None,
    inputgrid: Optional[interpolation.XGrid] = None,
):
    """Reinterpolate operators on output and/or input grids.

    The operation is in-place.

    Parameters
    ----------
    targetgrid : None or list
        xgrid for the target (output PDF)
    inputgrid : None or list
        xgrid for the input (input PDF)

    """
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
            eko.configs.interpolation_polynomial_degree,
            False,
        )
        target_rot = b.get_interpolation(targetgrid.raw)
        eko.rotations._targetgrid = targetgrid
    if inputgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            inputgrid,
            eko.configs.interpolation_polynomial_degree,
            False,
        )
        input_rot = b.get_interpolation(eko.rotations.inputgrid.raw)
        eko.rotations._inputgrid = inputgrid

    # build new grid
    for q2, elem in eko.items():
        ops = elem.operator
        errs = elem.error
        if targetgrid is not None and inputgrid is None:
            ops = np.einsum("ij,ajbk->aibk", target_rot, ops)
            errs = np.einsum("ij,ajbk->aibk", target_rot, errs)
        elif inputgrid is not None and targetgrid is None:
            ops = np.einsum("ajbk,kl->ajbl", ops, input_rot)
            errs = np.einsum("ajbk,kl->ajbl", errs, input_rot)
        else:
            ops = np.einsum("ij,ajbk,kl->aibl", target_rot, ops, input_rot)
            errs = np.einsum("ij,ajbk,kl->aibl", target_rot, errs, input_rot)
        elem.operator = ops
        elem.error = errs

        eko[q2] = elem


def flavor_reshape(
    eko: EKO,
    targetpids: Optional[np.ndarray] = None,
    inputpids: Optional[np.ndarray] = None,
):
    """Change the operators to have in the output targetpids and/or in the input inputpids.

    The operation is in-place.

    Parameters
    ----------
    targetpids : numpy.ndarray
        target rotation specified in the flavor basis
    inputpids : None or list
        input rotation specified in the flavor basis

    """
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
            ops = np.einsum("ca,ajbk->cjbk", targetpids, ops)
            errs = np.einsum("ca,ajbk->cjbk", targetpids, errs)
        elif inputpids is not None and targetpids is None:
            ops = np.einsum("ajbk,bd->ajdk", ops, inv_inputpids)
            errs = np.einsum("ajbk,bd->ajdk", errs, inv_inputpids)
        else:
            ops = np.einsum("ca,ajbk,bd->cjdk", targetpids, ops, inv_inputpids)
            errs = np.einsum("ca,ajbk,bd->cjdk", targetpids, errs, inv_inputpids)
        elem.operator = ops
        elem.error = errs

        eko[q2] = elem

    # drop PIDs - keeping them int nevertheless
    # there is no meaningful way to set them in general, after rotation
    if inputpids is not None:
        eko.rotations._inputpids = np.array([0] * len(eko.rotations.inputpids))
    if targetpids is not None:
        eko.rotations._targetpids = np.array([0] * len(eko.rotations.targetpids))


def to_evol(eko: EKO, source: bool = True, target: bool = False):
    """Rotate the operator into evolution basis.

    This also assigns also the pids. The operation is in-place.

    Parameters
    ----------
        source : bool
            rotate on the input tensor
        target : bool
            rotate on the output tensor

    """
    # rotate
    inputpids = br.rotate_flavor_to_evolution if source else None
    targetpids = br.rotate_flavor_to_evolution if target else None
    flavor_reshape(eko, inputpids=inputpids, targetpids=targetpids)
    # assign pids
    if source:
        eko.rotations._inputpids = br.evol_basis_pids
    if target:
        eko.rotations._targetpids = br.evol_basis_pids
