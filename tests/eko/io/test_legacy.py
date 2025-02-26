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

        # evolved PDFs at 10000 GeV^2 in the basis as in the lha benchmark
        pdf_test = {}
        for flav in [-6,-5,-4,-3,-2,-1,21,1,2,3,4,5,6]:
            pdf_test[flav] = evolved_pdfs[(Q2,4)][flav]

        pdfs = {}
        pdfs["u_v"] = pdf_test[2] - pdf_test[-2]
        pdfs["d_v"] = pdf_test[1] - pdf_test[-1]
        pdfs["L_p"] = 2*(pdf_test[-2] + pdf_test[-1])
        pdfs["L_m"] = pdf_test[-1] - pdf_test[-2]
        pdfs["s_p"] = pdf_test[3] + pdf_test[-3]
        pdfs["c_p"] = pdf_test[4] + pdf_test[-4]
        pdfs["g"] = pdf_test[21] 

        # Import the values of the LHA benchmark tables. This is not very nice yet, but should work
        lha_path = (
            pathlib.Path(__file__).parents[4]
            / "eko/src/ekomark/benchmark/external/LHA.yaml"
        )
        with open(lha_path, "r") as file:
            lha_benchmark = yaml.safe_load(file)

        pdf_benchmark = {}
        for key in ["u_v", "d_v", "L_p", "L_m", "s_p", "c_p", "g"]:
            globals()[f"pdf_benchmark_{key}"] = []
            globals()[f"xpdf_benchmark_{key}"] = lha_benchmark["table2"]["part2"][key]
            for i in range(len(x_grid)):
                globals()[f"pdf_benchmark_{key}"].append(globals()[f"xpdf_benchmark_{key}"][i] / x_grid[i])
            #globals()[f"pdf_benchmark_{key}"] = globals()[f"pdf_benchmark_{key}"][:-1]
            pdf_benchmark[key] = globals()[f"pdf_benchmark_{key}"]
            for j in range(len(pdfs[key])-5):
                np.testing.assert_allclose(pdfs[key][j], pdf_benchmark[key][j], rtol=0.0005) 
            for k in [6, 7, 8, 9, 10]:
                np.testing.assert_allclose(pdfs[key][k], pdf_benchmark[key][k], atol=1e-3)



test_read_legacy_cards()
test_read_legacy_pdf()

