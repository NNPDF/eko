"""Tools to evolve actual PDFs."""
import pathlib

import numpy as np

import eko
from eko import basis_rotation as br
from eko.io import legacy
from ekomark import apply

from . import genpdf, info_file


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
    eko_output = None
    if path is not None:
        my_path = pathlib.Path(path)
        if my_path.is_dir():
            outpath = my_path / ekofileid(theory_card, operators_card)
            eko_output = legacy.load_tar(outpath)
        else:
            eko_output = legacy.load_tar(my_path)
    else:
        eko_output = eko.solve(theory_card, operators_card)
        if store_path is not None:
            legacy.dump_tar(eko_output, store_path)

    evolved_PDF_list = []
    for initial_PDF in initial_PDF_list:
        evolved_PDF_list.append(apply.apply_pdf(eko_output, initial_PDF, targetgrid))

    if targetgrid is None:
        targetgrid = operators_card["rotations"]["xgrid"]
    if info_update is None:
        info_update = {}
    info_update["XMin"] = targetgrid[0]
    info_update["XMax"] = targetgrid[-1]
    info = info_file.build(
        theory_card,
        operators_card,
        len(evolved_PDF_list),
        info_update=info_update,
    )
    all_member_blocks = []
    for evolved_PDF in evolved_PDF_list:
        all_blocks = []
        block = genpdf.generate_block(
            lambda pid, x, Q2, evolved_PDF=evolved_PDF: targetgrid[targetgrid.index(x)]
            * evolved_PDF[Q2]["pdfs"][pid][targetgrid.index(x)],
            xgrid=targetgrid,
            Q2grid=operators_card["Q2grid"],
            pids=np.array(br.flavor_basis_pids),
        )
        # all_blocks will be useful in case there will be necessity to dump many blocks
        # for a single member
        all_blocks.append(block)
        all_member_blocks.append(all_blocks)

    genpdf.export.dump_set(name, info, all_member_blocks)

    if install:
        genpdf.install_pdf(name)


def ekofileid(theory_card, operators_card):
    """
    Return a common filename composed by the hashes.

    Parameters
    ----------
    theory_card : dict
        theory card
    operators_card : dict
        operators card

    Returns
    -------
    str
        file name
    """
    try:
        return f"o{operators_card['hash'][:6]}_t{theory_card['hash'][:6]}.tar"
    except KeyError:
        return "o000000_t000000.tar"
