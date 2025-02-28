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

TEST_DATA_DIR = pathlib.Path(__file__).parents[2] / "data"   # directory of the EKO object
pdf = toy.mkPDF("",0)

x_grid = LHA_utils.toy_xgrid
EP = (10000, 4)


def test_read_legacy():
    for name in ["v0.13.tar", "v0.14.tar", "v0.0.tar"]:
        with eko.EKO.read(TEST_DATA_DIR / name) as evolution_operator:
            th = evolution_operator.theory_card
            op = evolution_operator.operator_card
            assert isinstance(
                th, eko.io.runcards.TheoryCard
            )  # Check that theory card is read as theory card
            assert isinstance(
                op, eko.io.runcards.OperatorCard
            )  # Check that operator card is read as operator card

            
            # Check if the operator has the correct dimensions
            with evolution_operator.operator(EP) as op:
                if op.operator.shape != (14, 60, 14, 60):
                    print("Operator does not have the correct dimensions")

            # Use the operator on the pdf
            evolved_pdfs, _integration_errors = apply_pdf(
                evolution_operator, pdf, x_grid
            )

        # Import the values of the LHA benchmark tables
        lha_path = (
            pathlib.Path(__file__).parents[4]
            / "eko/src/ekomark/benchmark/external/LHA.yaml"
        )
        with open(lha_path, "r") as file:
            lha_benchmark = yaml.safe_load(file)

        # Make the test
        pdf_test = {}
        pdf_bench = {}
        for flav in [-6,-5,-4,-3,-2,-1,21,1,2,3,4,5,6]:
            pdf_test[flav] = evolved_pdfs[EP][flav]
            pdf_bench[flav] = LHA_utils.rotate_data(lha_benchmark["table2"]["part2"])[flav]
            np.testing.assert_allclose(pdf_test[flav], (pdf_bench[flav]/x_grid), rtol=5e-4, atol=5e-6)




