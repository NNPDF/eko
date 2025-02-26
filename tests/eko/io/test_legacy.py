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
import ekomark.benchmark.runner

TEST_DATA_DIR = pathlib.Path(__file__).parents[2] / "data"   # directory of the EKO object
pdf = toy.mkPDF("",0)

x_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9]
Q2 = 10000


def test_read_legacy_cards():
    for name in ["v0.13.5.tar", "v0.14.tar", "v0.0.tar"]:
        with eko.EKO.read(TEST_DATA_DIR / name) as evolution_operator:
            # import pdb; pdb.set_trace()
            assert isinstance(
                evolution_operator.theory_card, eko.io.runcards.TheoryCard
            )  # Check that theory card is read as theory card
            assert isinstance(
                evolution_operator.operator_card, eko.io.runcards.OperatorCard
            )  # Check that operator card is read as operator card

def test_read_legacy_pdf():
    for name in ["v0.13.5.tar", "v0.14.tar", "v0.0.tar"]:
        with eko.EKO.read(TEST_DATA_DIR / name) as evolution_operator:
            
            # Check if the operator has the correct dimensions
            with evolution_operator.operator((10000.0, 4)) as op:
                if op.operator.shape != (14, 60, 14, 60):
                    print("Operator does not have the correct dimensions")

            # Use the operator on the pdf
            evolved_pdfs, _integration_errors = apply_pdf(
                evolution_operator, pdf, x_grid
            )

        print("Evolution grid", evolution_operator.evolgrid)
        pdf_test = evolved_pdfs[(Q2,4)][21]  # evolved gluon PDF at 10000 GeV^2
        print("Gluon pdf:", pdf_test)

        # Import the values of the LHA benchmark tables. This is not very nice yet, but should work
        lha_path = (
            pathlib.Path(__file__).parents[4]
            / "eko/src/ekomark/benchmark/external/LHA.yaml"
        )
        with open(lha_path, "r") as file:
            lha_benchmark = yaml.safe_load(file)

        xpdf_benchmark = lha_benchmark["table2"]["part2"]["g"]

        pdf_benchmark = []  # gluon PDF at 10000 GeV^2 from the LHA benchmark tables
        
        for j in range(len(xpdf_benchmark)):
            pdf_benchmark.append(xpdf_benchmark[j] / x_grid[j]) # have to divide by x values to compare

        
        # Test that the PDF values are the same, taking a relative tolerance of 0.09
        
        np.testing.assert_allclose(pdf_test, pdf_benchmark, rtol=0.09, atol=172050)
        

    return evolved_pdfs

test_read_legacy_cards()
test_read_legacy_pdf()

