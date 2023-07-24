"""Apply operator evolution to PDF set."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from eko import basis_rotation as br
from eko import interpolation
from eko.io import EKO
from eko.io.types import EvolutionPoint


def apply_pdf(
    eko: EKO,
    lhapdf_like,
    targetgrid=None,
    rotate_to_evolution_basis=False,
    qed=False,
):
    """
    Apply all available operators to the input PDFs.

    Parameters
    ----------
        output : eko.output.EKO
            eko output object containing all informations
        lhapdf_like : object
            object that provides an xfxQ2 callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
            and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
        targetgrid : list
            if given, interpolates to the pdfs given at targetgrid (instead of xgrid)
        rotate_to_evolution_basis : bool
            if True rotate to evoluton basis

    Returns
    -------
        out_grid : dict
            output PDFs and their associated errors for the computed mu2grid
    """
    if rotate_to_evolution_basis:
        if not qed:
            rotate_flavor_to_evolution = br.rotate_flavor_to_evolution
        else:
            rotate_flavor_to_evolution = br.rotate_flavor_to_unified_evolution
        return apply_pdf_flavor(
            eko, lhapdf_like, targetgrid, rotate_flavor_to_evolution, qed
        )
    return apply_pdf_flavor(eko, lhapdf_like, targetgrid)


CONTRACTION = "ajbk,bk"


@dataclass
class PdfResult:
    """Helper class to collect PDF results."""

    pdfs: Dict[Union[int, str], float]
    errors: Optional[Dict[Union[int, str], float]] = None


def apply_pdf_flavor(
    eko: EKO, lhapdf_like, targetgrid=None, flavor_rotation=None, qed=False
):
    """
    Apply all available operators to the input PDFs.

    Parameters
    ----------
        output : eko.output.EKO
            eko output object containing all informations
        lhapdf_like : object
            object that provides an xfxQ2 callable (as `lhapdf <https://lhapdf.hepforge.org/>`_
            and :class:`ekomark.toyLH.toyPDF` do) (and thus is in flavor basis)
        targetgrid : list
            if given, interpolates to the pdfs given at targetgrid (instead of xgrid)
        flavor_rotation : np.ndarray
            Rotation matrix in flavor space
        qed : bool
            activate qed

    Returns
    -------
        out_grid : dict
            output PDFs and their associated errors for the computed mu2grid
    """
    # create pdfs
    pdfs = np.zeros((len(eko.bases.inputpids), len(eko.bases.inputgrid)))
    for j, pid in enumerate(eko.bases.inputpids):
        if not lhapdf_like.hasFlavor(pid):
            continue
        pdfs[j] = np.array(
            [lhapdf_like.xfxQ2(pid, x, eko.mu20) / x for x in eko.bases.inputgrid.raw]
        )

    # build output
    out_grid: Dict[EvolutionPoint, PdfResult] = {}
    for ep, elem in eko.items():
        pdf_final = np.einsum(CONTRACTION, elem.operator, pdfs, optimize="optimal")
        if elem.error is not None:
            error_final = np.einsum(CONTRACTION, elem.error, pdfs, optimize="optimal")
        else:
            error_final = None
        out_grid[ep] = PdfResult(dict(zip(eko.bases.targetpids, pdf_final)))
        if error_final is not None:
            out_grid[ep].errors = dict(zip(eko.bases.targetpids, error_final))

    # rotate to evolution basis
    if flavor_rotation is not None:
        for q2, op in out_grid.items():
            pdf = flavor_rotation @ np.array(
                [op.pdfs[pid] for pid in br.flavor_basis_pids]
            )
            if not qed:
                evol_basis = br.evol_basis
            else:
                evol_basis = br.unified_evol_basis
            op.pdfs = dict(zip(evol_basis, pdf))
            if op.errors is not None:
                errors = flavor_rotation @ np.array(
                    [op.errors[pid] for pid in br.flavor_basis_pids]
                )
                op.errors = dict(zip(evol_basis, errors))

    # rotate/interpolate to target grid
    if targetgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            xgrid=eko.bases.targetgrid,
            polynomial_degree=eko.operator_card.configs.interpolation_polynomial_degree,
            mode_N=False,
        )

        rot = b.get_interpolation(targetgrid)
        for evpdf in out_grid.values():
            for pdf_label in evpdf.pdfs:
                evpdf.pdfs[pdf_label] = np.matmul(rot, evpdf.pdfs[pdf_label])
                if evpdf.errors is not None:
                    evpdf.errors[pdf_label] = np.matmul(rot, evpdf.errors[pdf_label])
    # cast back to be backward compatible
    real_out_grid = {}
    for ep, res in out_grid.items():
        real_out_grid[ep] = {"pdfs": res.pdfs, "errors": res.errors}
    return real_out_grid
