from banana.data import genpdf

import eko
from eko import basis_rotation as br

from . import apply, gen_op, gen_theory, lhapdf_style


def evolve_PDFs(
    initial_PDF,
    theory_card,
    operators_card,
    targetgrid=None,
    install=False,
    name="Evolved_PDF",
    info_update=None,
):
    """
    This function evolves an initial_PDF using a theory card and an operator card
    and dump the evolved PDF in lhapdf format

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

        install : bool
            set whether to install evolved PDF to lhapdf directory

        name : str
            set name of evolved PDF

        info_update : dict
            dict of info to add or update to default info file

    """
    eko_output = eko.run_dglap(theory_card, operators_card)

    evolved_PDF = apply.apply_pdf(eko_output, initial_PDF, targetgrid)
    # TODOs: What will happen with more than one members? (iterate, change apply)
    # TODOs: Consider a different number of flavors according to theory card ("this is not a problem right now")
    # to solve it I need to implement a function optimize to check for zeros

    if targetgrid is None:
        targetgrid = operators_card["interpolation_xgrid"]

    info = lhapdf_style.create_info_file(theory_card, operators_card, info_update)
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

    if install:
        genpdf.install_pdf(name)
