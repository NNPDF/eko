import pathlib
from math import nan
import yaml
from banana import toy

import eko

import numpy as np
from eko.interpolation import make_grid
from ekobox.evol_pdf import evolve_pdfs
import lhapdf
from ekobox.apply import apply_pdf
import numpy.testing
from numpy.testing import assert_almost_equal
from ekomark.benchmark.runner import Runner
from ekomark.benchmark.external import LHA_utils
import eko.basis_rotation as br

TEST_DATA_DIR = pathlib.Path(__file__).parents[2] / "data"   # directory of the EKO object
pdf = toy.mkPDF("",0)

x_grid = LHA_utils.toy_xgrid
EP = (10000., 4)


def test_read_legacy():
    for name in ["v0.13.tar", "v0.14.tar", "v0.0.tar"]:
        with eko.EKO.read(TEST_DATA_DIR / name) as evolution_operator:
            assert isinstance(
                evolution_operator.theory_card, eko.io.runcards.TheoryCard
            )  # Check that theory card is read as theory card
            assert isinstance(
                evolution_operator.operator_card, eko.io.runcards.OperatorCard
            )  # Check that operator card is read as operator card

            
            # Check if the operator has the correct dimensions
            with evolution_operator.operator(EP) as op:
                assert op.operator.shape == (14, 60, 14, 60)
                    

            # Use the operator on the pdf
            evolved_pdfs, _integration_errors = apply_pdf(
                evolution_operator, pdf, x_grid
            )

        # Import the values of the LHA benchmark tables
        lha_benchmark = LHA_utils.compute_LHA_data(
            {"FNS": "FFNS", "PTO": 0, "XIF": 1},
            {"polarized": 0, "mugrid": 100}
        )
        
        # Make the test
        pdf_test = {}
        pdf_bench = {}
        for flav in br.flavor_basis_pids:
            pdf_test[flav] = evolved_pdfs[EP][flav]
            pdf_bench[flav] = lha_benchmark["values"][EP[0]][flav]
            np.testing.assert_allclose(pdf_test[flav], (pdf_bench[flav]/x_grid), rtol=5e-4, atol=5e-6)




