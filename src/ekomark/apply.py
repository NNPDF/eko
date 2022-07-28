# -*- coding: utf-8 -*-
import numpy as np

from eko import basis_rotation as br
from eko import interpolation


def apply_pdf(eko, lhapdf_like, targetgrid=None, rotate_to_evolution_basis=False):
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
            output PDFs and their associated errors for the computed Q2grid
    """
    if rotate_to_evolution_basis:
        return apply_pdf_flavor(
            eko, lhapdf_like, targetgrid, br.rotate_flavor_to_evolution
        )
    return apply_pdf_flavor(eko, lhapdf_like, targetgrid)


def apply_pdf_flavor(eko, lhapdf_like, targetgrid=None, flavor_rotation=None):
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

    Returns
    -------
        out_grid : dict
            output PDFs and their associated errors for the computed Q2grid
    """
    # create pdfs
    pdfs = np.zeros((len(eko.rotations.inputpids), len(eko.rotations.inputgrid)))
    for j, pid in enumerate(eko.rotations.inputpids):
        if not lhapdf_like.hasFlavor(pid):
            continue
        pdfs[j] = np.array(
            [
                lhapdf_like.xfxQ2(pid, x, eko.Q02) / x
                for x in eko.rotations.inputgrid.raw
            ]
        )

    # build output
    out_grid = {}
    for q2, elem in eko.items():
        pdf_final = np.einsum("ajbk,bk", elem.operator, pdfs)
        error_final = np.einsum("ajbk,bk", elem.error, pdfs)
        out_grid[q2] = {
            "pdfs": dict(zip(eko.rotations.targetpids, pdf_final)),
            "errors": dict(zip(eko.rotations.targetpids, error_final)),
        }

    # rotate to evolution basis
    if flavor_rotation is not None:
        for q2, op in out_grid.items():
            pdf = flavor_rotation @ np.array(
                [op["pdfs"][pid] for pid in br.flavor_basis_pids]
            )
            errors = flavor_rotation @ np.array(
                [op["errors"][pid] for pid in br.flavor_basis_pids]
            )
            op["pdfs"] = dict(zip(br.evol_basis, pdf))
            op["errors"] = dict(zip(br.evol_basis, errors))

    # rotate/interpolate to target grid
    if targetgrid is not None:
        b = eko.interpolator(False, True)
        rot = b.get_interpolation(targetgrid)
        for evpdf in out_grid.values():
            for pdf_label in evpdf["pdfs"]:
                evpdf["pdfs"][pdf_label] = np.matmul(rot, evpdf["pdfs"][pdf_label])
                evpdf["errors"][pdf_label] = np.matmul(rot, evpdf["errors"][pdf_label])

    return out_grid
