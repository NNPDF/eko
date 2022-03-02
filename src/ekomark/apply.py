# -*- coding: utf-8 -*-
import numpy as np

from eko import basis_rotation as br
from eko import interpolation


def apply_pdf(output, lhapdf_like, targetgrid=None, rotate_to_evolution_basis=False):
    """
    Apply all available operators to the input PDFs.

    Parameters
    ----------
        output : eko.output.Output
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
            output, lhapdf_like, targetgrid, br.rotate_flavor_to_evolution
        )
    return apply_pdf_flavor(output, lhapdf_like, targetgrid)


def apply_pdf_flavor(output, lhapdf_like, targetgrid=None, flavor_rotation=None):
    """
    Apply all available operators to the input PDFs.

    Parameters
    ----------
        output : eko.output.Output
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
    pdfs = np.zeros((len(output["inputpids"]), len(output["inputgrid"])))
    for j, pid in enumerate(output["inputpids"]):
        if not lhapdf_like.hasFlavor(pid):
            continue
        pdfs[j] = np.array(
            [
                lhapdf_like.xfxQ2(pid, x, output["q2_ref"]) / x
                for x in output["inputgrid"]
            ]
        )

    # build output
    out_grid = {}
    for q2, elem in output["Q2grid"].items():
        pdf_final = np.einsum("ajbk,bk", elem["operators"], pdfs)
        error_final = np.einsum("ajbk,bk", elem["operator_errors"], pdfs)
        out_grid[q2] = {
            "pdfs": dict(zip(output["targetpids"], pdf_final)),
            "errors": dict(zip(output["targetpids"], error_final)),
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
            out_grid[q2]["pdfs"] = dict(zip(br.evol_basis, pdf))
            out_grid[q2]["errors"] = dict(zip(br.evol_basis, errors))

    # rotate/interpolate to target grid
    if targetgrid is not None:
        b = interpolation.InterpolatorDispatcher.from_dict(output, False)
        rot = b.get_interpolation(targetgrid)
        for q2 in out_grid:
            for pdf_label in out_grid[q2]["pdfs"]:
                out_grid[q2]["pdfs"][pdf_label] = np.matmul(
                    rot, out_grid[q2]["pdfs"][pdf_label]
                )
                out_grid[q2]["errors"][pdf_label] = np.matmul(
                    rot, out_grid[q2]["errors"][pdf_label]
                )

    return out_grid
