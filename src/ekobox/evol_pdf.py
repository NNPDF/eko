"""Tools to evolve actual PDFs."""
import pathlib
from collections import defaultdict

import numpy as np

import eko
from eko import basis_rotation as br
from eko.io import EKO

from . import apply, genpdf, info_file

DEFAULT_NAME = "eko.tar"


def evolve_pdfs(
    initial_PDF_list,
    theory_card,
    operators_card,
    path=None,
    store_path=None,
    targetgrid=None,
    install=False,
    name="Evolved_PDF",
    info_update=None,
):
    """Evolves one or several initial_PDFs and dump the evolved PDFs in lhapdf format.

    Parameters
    ----------
    initial_PDF_list : list(lhapdf object)
        list of PDF members to be evolved
    theory_card : dict
        theory card
    operators_card : dict
        operators card
    path : str
        path to cached eko output (if "None" it will be recomputed)
    store_path : str
        path where the eko is stored (if "None" will not be saved)
    targetgrid : list(float)
        target x-grid (if different from input x-grid)
    install : bool
        set whether to install evolved PDF to lhapdf directory
    name : str
        set name of evolved PDF
    info_update : dict
        dict of info to add or update to default info file
    """
    # update op and th cards
    if path is not None:
        eko_path = pathlib.Path(path)
        if eko_path.is_dir():
            eko_path = eko_path / DEFAULT_NAME
    else:
        if store_path is None:
            raise ValueError("'store_path' required if 'path' is not provided.")
        eko.solve(theory_card, operators_card, path=store_path)
        eko_path = store_path

    # apply PDF to eko
    evolved_PDF_list = []
    with EKO.read(eko_path) as eko_output:
        for initial_PDF in initial_PDF_list:
            evolved_PDF_list.append(
                apply.apply_pdf(eko_output, initial_PDF, targetgrid)
            )

    # update info file
    if targetgrid is None:
        targetgrid = operators_card.xgrid
    if info_update is None:
        info_update = {}
    info_update["XMin"] = targetgrid.raw[0]
    info_update["XMax"] = targetgrid.raw[-1]
    info = info_file.build(
        theory_card,
        operators_card,
        len(evolved_PDF_list),
        info_update=info_update,
    )

    # separate by nf the evolgrid (and order per nf/q)
    q2block_per_nf = regroup_evolgrid(eko_output.evolgrid)

    # write all replicas
    all_member_blocks = []
    for evolved_PDF in evolved_PDF_list:
        all_blocks = collect_blocks(
            evolved_PDF, q2block_per_nf, targetgrid.raw.tolist()
        )
        all_member_blocks.append(all_blocks)
    genpdf.export.dump_set(name, info, all_member_blocks)

    if install:
        genpdf.install_pdf(name)


def regroup_evolgrid(evolgrid: list):
    """Split evolution points by nf and sort by scale."""
    by_nf = defaultdict(list)
    for q, nf in sorted(evolgrid, key=lambda ep: ep[1]):
        by_nf[nf].append(q)
    return {nf: sorted(qs) for nf, qs in by_nf.items()}


def collect_blocks(evolved_PDF: dict, q2block_per_nf: dict, xgrid: list):
    """Collect all LHAPDF blocks for a given replica.

    Parameters
    ----------
    evolved_PDF :
        PDF evaluated at grid
    q2block_per_nf :
        block coordinates
    xgrid :
        x grid

    """
    all_blocks = []

    # fake xfxQ2
    def pdf_xq2(pid, x, Q2):
        x_idx = xgrid.index(x)
        return x * evolved_PDF[(Q2, nf)]["pdfs"][pid][x_idx]

    # loop on nf patches
    for nf, q2grid in q2block_per_nf.items():
        block = genpdf.generate_block(
            pdf_xq2,
            xgrid=xgrid,
            sorted_q2grid=q2grid,
            pids=br.flavor_basis_pids,
        )
        all_blocks.append(block)
    return all_blocks
