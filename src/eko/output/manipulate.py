# -*- coding: utf-8 -*-
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
    targetgrid: Optional[np.ndarray] = None,
    inputgrid: Optional[np.ndarray] = None,
    inplace: bool = True,
):
    """Change the operators to have in the output targetgrid and/or in the input inputgrid.

    The operation is in-place.

    Parameters
    ----------
    targetgrid : None or list
        xgrid for the target
    inputgrid : None or list
        xgrid for the input

    """
    # calling with no arguments is an error
    if targetgrid is None and inputgrid is None:
        raise ValueError("Nor inputgrid nor targetgrid was given")
    # now check to the current status
    if (
        targetgrid is not None
        and len(targetgrid) == len(eko.rotations.targetgrid)
        and np.allclose(targetgrid, eko.rotations.targetgrid)
    ):
        targetgrid = None
        warnings.warn("The new targetgrid is close to the current targetgrid")
    if (
        inputgrid is not None
        and len(inputgrid) == len(eko.rotations.inputgrid)
        and np.allclose(inputgrid, eko.rotations.inputgrid)
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
            eko.configs.interpolation_is_log,
            False,
        )
        target_rot = b.get_interpolation(targetgrid)
        eko.rotations.targetgrid = np.array(targetgrid)
    if inputgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            inputgrid,
            eko.configs.interpolation_polynomial_degree,
            eko.configs.interpolation_is_log,
            False,
        )
        input_rot = b.get_interpolation(eko.rotations.inputgrid)
        eko.rotations.inputgrid = np.array(inputgrid)

    # build new grid
    for _, elem in eko.items():
        if elem is None:
            continue
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


def flavor_reshape(
    eko: EKO,
    targetbasis: Optional[np.ndarray] = None,
    inputbasis: Optional[np.ndarray] = None,
    inplace: bool = True,
):
    """Change the operators to have in the output targetbasis and/or in the input inputbasis.

    The operation is in-place.

    Parameters
    ----------
    targetbasis : numpy.ndarray
        target rotation specified in the flavor basis
    inputbasis : None or list
        input rotation specified in the flavor basis

    """
    # calling with no arguments is an error
    if targetbasis is None and inputbasis is None:
        raise ValueError("Nor inputbasis nor targetbasis was given")
    # now check to the current status
    if targetbasis is not None and np.allclose(
        targetbasis, np.eye(len(eko.rotations.targetpids))
    ):
        targetbasis = None
        warnings.warn("The new targetbasis is close to current basis")
    if inputbasis is not None and np.allclose(
        inputbasis, np.eye(len(eko.rotations.inputpids))
    ):
        inputbasis = None
        warnings.warn("The new inputbasis is close to current basis")
    # after the checks: if there is still nothing to do, skip
    if targetbasis is None and inputbasis is None:
        logger.debug("Nothing done.")
        return

    # flip input around
    if inputbasis is not None:
        inv_inputbasis = np.linalg.inv(inputbasis)

    # build new grid
    for _, elem in eko.items():
        if elem is None:
            continue
        ops = elem.operator
        errs = elem.error
        if targetbasis is not None and inputbasis is None:
            ops = np.einsum("ca,ajbk->cjbk", targetbasis, ops)
            errs = np.einsum("ca,ajbk->cjbk", targetbasis, errs)
        elif inputbasis is not None and targetbasis is None:
            ops = np.einsum("ajbk,bd->ajdk", ops, inv_inputbasis)
            errs = np.einsum("ajbk,bd->ajdk", errs, inv_inputbasis)
        else:
            ops = np.einsum("ca,ajbk,bd->cjdk", targetbasis, ops, inv_inputbasis)
            errs = np.einsum("ca,ajbk,bd->cjdk", targetbasis, errs, inv_inputbasis)
        elem.operator = ops
        elem.error = errs
    # drop PIDs - keeping them int nevertheless
    #  there is no meaningful way to set them in general, after rotation
    if inputbasis is not None:
        eko.rotations.inputpids = [0] * len(eko.rotations.inputpids)
    if targetbasis is not None:
        eko.rotations.targetpids = [0] * len(eko.rotations.targetpids)


def to_evol(eko: EKO, source: bool = True, target: bool = False, inplace: bool = True):
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
    inputbasis = br.rotate_flavor_to_evolution if source else None
    targetbasis = br.rotate_flavor_to_evolution if target else None
    flavor_reshape(eko, inputbasis=inputbasis, targetbasis=targetbasis)
    # assign pids
    if source:
        eko.rotations.inputpids = br.evol_basis_pids
    if target:
        eko.rotations.targetpids = br.evol_basis_pids
