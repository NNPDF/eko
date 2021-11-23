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
):
    """
    This function evolves an initial_PDF using a theory card and an operator card
    and return the new PDF as lhapdf format

    """
    eko_output = eko.run_dglap(theory_card, operators_card)

    evolved_PDF = apply.apply_pdf_flavor(
        eko_output, initial_PDF, targetgrid, flavor_rotation
    )

    return evolved_PDF
