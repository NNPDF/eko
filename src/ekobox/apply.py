"""Apply evolution operator to a PDF."""

from collections.abc import Sequence
from typing import Optional

import numpy as np
import numpy.typing as npt

from eko import basis_rotation as br
from eko import interpolation
from eko.io import EKO
from eko.io.types import EvolutionPoint

RawPdfResult = dict[EvolutionPoint, npt.ArrayLike]
"""PDFs as raw grids.

The key is given by the associated evolution point. The values are
tensors sorted by (replica, flavor, xgrid). It may be the PDF or the
associated integration error.
"""


LabeledPdfResult = dict[EvolutionPoint, dict[int, npt.ArrayLike]]
"""PDFs labeled by their PDF identifier.

The outer key is given by the associated evolution point. The inner key
is the |PID|. The inner values are the values for along the xgrid. It
may be the PDF or the associated integration error.
"""


def apply_pdf(
    eko: EKO,
    lhapdf_like,
    targetgrid: npt.ArrayLike = None,
    rotate_to_evolution_basis: bool = False,
) -> tuple[LabeledPdfResult, LabeledPdfResult]:
    """Apply all available operators to the input PDF.

    Parameters
    ----------
    eko :
        eko output object containing all informations
    lhapdf_like : Any
        object that provides an `xfxQ2` callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
        and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
    targetgrid :
        if given, interpolates to the targetgrid (instead of xgrid)
    rotate_to_evolution_basis :
        if True rotate to evoluton basis

    Returns
    -------
    pdfs :
        PDFs for the computed evolution points
    errors :
        Integration errors for PDFs for the computed evolution points
    """
    # prepare post-process
    qed = eko.theory_card.order[1] > 0
    flavor_rotation = None
    labels = br.flavor_basis_pids
    if rotate_to_evolution_basis:
        if not qed:
            flavor_rotation = br.rotate_flavor_to_evolution
            labels = br.evol_basis_pids
        else:
            flavor_rotation = br.rotate_flavor_to_unified_evolution
            labels = br.unified_evol_basis_pids
    return apply_pdf_flavor(eko, lhapdf_like, labels, targetgrid, flavor_rotation)


def apply_pdf_flavor(
    eko: EKO,
    lhapdf_like,
    flavor_labels: Sequence[int],
    targetgrid: npt.ArrayLike = None,
    flavor_rotation: npt.ArrayLike = None,
) -> tuple[LabeledPdfResult, LabeledPdfResult]:
    """Apply all available operators to the input PDF.

    Parameters
    ----------
    eko :
        eko output object containing all informations
    lhapdf_like : Any
        object that provides an `xfxQ2` callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
        and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
    flavor_labels :
        flavor names
    targetgrid :
        if given, interpolates to the targetgrid (instead of xgrid)
    flavor_rotation :
        if give, rotate in flavor space

    Returns
    -------
    pdfs :
        PDFs for the computed evolution points
    errors :
        Integration errors for PDFs for the computed evolution points
    """
    # create pdfs
    input_pdfs = np.zeros((len(br.flavor_basis_pids), len(eko.xgrid)))
    for j, pid in enumerate(br.flavor_basis_pids):
        if not lhapdf_like.hasFlavor(pid):
            continue
        input_pdfs[j] = np.array(
            [lhapdf_like.xfxQ2(pid, x, eko.mu20) / x for x in eko.xgrid.raw]
        )
    # apply
    grids, grid_errors = apply_grids(eko, input_pdfs[None, :])
    new_grids = rotate_result(eko, grids, flavor_labels, targetgrid, flavor_rotation)
    new_errors = rotate_result(
        eko, grid_errors, flavor_labels, targetgrid, flavor_rotation
    )
    # unwrap the replica axis again
    pdfs: LabeledPdfResult = {}
    errors: LabeledPdfResult = {}
    for ep, pdf in new_grids.items():
        pdfs[ep] = {lab: grid[0] for lab, grid in pdf.items()}
        if ep in new_errors:
            errors[ep] = {lab: (grid[0]) for lab, grid in new_errors[ep].items()}
    return pdfs, errors


def rotate_result(
    eko: EKO,
    grids: RawPdfResult,
    flavor_labels: Sequence[int],
    targetgrid: Optional[npt.ArrayLike] = None,
    flavor_rotation: Optional[npt.ArrayLike] = None,
) -> LabeledPdfResult:
    """Rotate and relabel PDFs.

    Parameters
    ----------
    eko :
        eko output object containing all informations
    grids :
        Raw grids coming from evolution
    flavor_labels :
        flavors names
    targetgrid :
        if given, interpolates to the targetgrid (instead of xgrid)
    flavor_rotation :
        if given, rotates in flavor space

    Returns
    -------
    pdfs :
        relabeled and, if requested rotated, PDFs
    """
    # rotate to evolution basis
    if flavor_rotation is not None:
        new_grids = {}
        for ep, pdf_grid in grids.items():
            new_grids[ep] = np.einsum(
                "ab,rbk->rak", flavor_rotation, pdf_grid, optimize="optimal"
            )
        grids = new_grids

    # rotate/interpolate to target grid
    if targetgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            xgrid=eko.xgrid,
            polynomial_degree=eko.operator_card.configs.interpolation_polynomial_degree,
            mode_N=False,
        )

        x_rotation = b.get_interpolation(targetgrid)
        new_grids = {}
        for ep, pdf_grid in grids.items():
            new_grids[ep] = np.einsum(
                "jk,rbk->rbj", x_rotation, pdf_grid, optimize="optimal"
            )
        grids = new_grids

    # relabel
    new_grids = {}
    for ep, pdf_grid in grids.items():
        new_grids[ep] = dict(
            zip(
                flavor_labels,
                np.swapaxes(pdf_grid, 0, 1),
            )
        )
    grids = new_grids

    return grids


_EKO_CONTRACTION = "ajbk,rbk->raj"
"""Contract eko for all replicas."""


def apply_grids(
    eko: EKO, input_grids: npt.ArrayLike
) -> tuple[RawPdfResult, RawPdfResult]:
    """Apply all available operators to the input grids.

    Parameters
    ----------
    eko :
        eko output object
    input_grids :
        3D PDF grids evaluated at the inital scale. The axis have to be (replica, flavor, xgrid)

    Returns
    -------
    pdfs :
        output PDFs for the computed evolution points
    errors :
        associated integration errors for the computed evolution points
    """
    # sanity check
    if len(input_grids.shape) != 3 or input_grids.shape[1:] != (
        len(br.flavor_basis_pids),
        len(eko.xgrid),
    ):
        raise ValueError(
            "input grids have to be sorted by replica, flavor, xgrid!"
            f"The shape has to be (r,{len(br.flavor_basis_pids)},{len(eko.xgrid)})"
        )
    # iterate
    pdfs: RawPdfResult = {}
    errors: RawPdfResult = {}
    for ep, elem in eko.items():
        pdfs[ep] = np.einsum(
            _EKO_CONTRACTION, elem.operator, input_grids, optimize="optimal"
        )
        if elem.error is not None:
            errors[ep] = np.einsum(
                _EKO_CONTRACTION, elem.error, input_grids, optimize="optimal"
            )
    return pdfs, errors
