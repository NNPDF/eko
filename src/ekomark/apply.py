"""Apply operator evolution to PDF set."""

import numpy as np

from eko import basis_rotation as br
from eko import interpolation
from eko.io import EKO


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
            output PDFs and their associated errors for the computed Q2grid
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
            output PDFs and their associated errors for the computed Q2grid
    """
    # create pdfs
    pdfs = np.zeros((len(eko.rotations.inputpids), len(eko.rotations.inputgrid)))
    for j, pid in enumerate(eko.rotations.inputpids):
        if not lhapdf_like.hasFlavor(pid):
            continue
        pdfs[j] = np.array(
            [
                lhapdf_like.xfxQ2(pid, x, eko.mu20) / x
                for x in eko.rotations.inputgrid.raw
            ]
        )

    # build output
    out_grid = {}
    for mu2, elem in eko.items():
        pdf_final = np.einsum(CONTRACTION, elem.operator, pdfs)
        if elem.error is not None:
            error_final = np.einsum(CONTRACTION, elem.error, pdfs)
        else:
            error_final = None
        out_grid[mu2] = {
            "pdfs": dict(zip(eko.rotations.targetpids, pdf_final)),
            "errors": None,
        }
        if error_final is not None:
            out_grid[mu2]["errors"] = dict(zip(eko.rotations.targetpids, error_final))

    # rotate to evolution basis
    if flavor_rotation is not None:
        for q2, op in out_grid.items():
            pdf = flavor_rotation @ np.array(
                [op["pdfs"][pid] for pid in br.flavor_basis_pids]
            )
            if not qed:
                evol_basis = br.evol_basis
            else:
                evol_basis = br.unified_evol_basis
            op["pdfs"] = dict(zip(evol_basis, pdf))
            if op["errors"] is not None:
                errors = flavor_rotation @ np.array(
                    [op["errors"][pid] for pid in br.flavor_basis_pids]
                )
                op["errors"] = dict(zip(evol_basis, errors))

    # rotate/interpolate to target grid
    if targetgrid is not None:
        b = interpolation.InterpolatorDispatcher(
            xgrid=eko.rotations.targetgrid,
            polynomial_degree=eko.operator_card.configs.interpolation_polynomial_degree,
            mode_N=False,
        )

        rot = b.get_interpolation(targetgrid)
        for evpdf in out_grid.values():
            for pdf_label in evpdf["pdfs"]:
                evpdf["pdfs"][pdf_label] = np.matmul(rot, evpdf["pdfs"][pdf_label])
                if evpdf["errors"] is not None:
                    evpdf["errors"][pdf_label] = np.matmul(
                        rot, evpdf["errors"][pdf_label]
                    )

    return out_grid
