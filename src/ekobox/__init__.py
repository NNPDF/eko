# -*- coding: utf-8 -*-
from banana.data import genpdf

import eko
from eko import basis_rotation as br

from . import apply, gen_op, gen_theory


# TODO: give the possibility to dump and/or install in lhapdf folder (using genpdf package if possible)
def evolve_PDFs(
    initial_PDF,
    theory_card,
    operators_card,
    targetgrid=None,
    flavor_rotation=None,
    dump=False,
    install=False,
    name=None,
    info_update=None,
):
    """
    This function evolves an initial_PDF using a theory card and an operator card
    and return the new PDF as lhapdf format

    Parameters
    ----------
        initial_PDF : lhapdf object
            PDF to be evolved

        theory_card : dict
            theory card

        operators_card : dict
            operators card

        targetgrid : list(float)
            target x-grid (if different from input x-grid)

        flavor_rotation : np.ndarray
            Rotation matrix in flavor space

        dump : bool
            set whether to dump evolved PDF to current dicrectory

        install : bool
            set whether to install evolved PDF to lhapdf directory

        name : str
            set name of evolved PDF (if dumped)

        info_update : dict
            dict of info to add or update to default info file (if PDF is dumped)

    Returns
    -------
            : dict
            output PDFs and their associated errors for the computed Q2grid

    """
    eko_output = eko.run_dglap(theory_card, operators_card)

    evolved_PDF = apply.apply_pdf_flavor(
        eko_output, initial_PDF, targetgrid, flavor_rotation
    )
    # TODOs: What will happen with more than one members?
    # TODOs: Consider a different number of flavors according to theory card
    # TODOs: Consider the flavor_rotation case
    if dump:
        if targetgrid is None:
            targetgrid = operators_card["interpolation_xgrid"]
        if name is None:
            name = "Evolved_PDF"
        info = gen_theory.create_info_file(theory_card, operators_card, info_update)
        block = genpdf.generate_block(
            lambda pid, x, Q2: evolved_PDF[Q2]["pdfs"][pid][targetgrid.index(x)],
            xgrid=targetgrid,
            Q2grid=operators_card["Q2grid"],
            pids=br.flavor_basis_pids,
        )
        error_block = genpdf.generate_block(
            lambda pid, x, Q2: evolved_PDF[Q2]["errors"][pid][targetgrid.index(x)],
            xgrid=targetgrid,
            Q2grid=operators_card["Q2grid"],
            pids=br.flavor_basis_pids,
        )

        genpdf.export.dump_set(name, info, [[block], [error_block]])

    return evolved_PDF
