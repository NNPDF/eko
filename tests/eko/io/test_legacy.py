import pathlib

import numpy as np
import numpy.testing
from banana import toy

import eko
import eko.basis_rotation as br
from ekobox.apply import apply_pdf
from ekomark.benchmark.external import LHA_utils

TEST_DATA_DIR = (
    pathlib.Path(__file__).parents[2] / "data"
)  # directory of the EKO object
pdf = toy.mkPDF("", 0)

x_grid = LHA_utils.toy_xgrid
EP = (10000.0, 4)


def test_read_legacy():
    for name in ["v0.13.tar", "v0.14.tar", "v0.15.tar"]:
        with eko.EKO.read(TEST_DATA_DIR / name) as evolution_operator:
            # Check the cards
            assert isinstance(
                evolution_operator.theory_card, eko.io.runcards.TheoryCard
            )
            assert isinstance(
                evolution_operator.operator_card, eko.io.runcards.OperatorCard
            )

            # Check if the operator has the correct dimensions
            with evolution_operator.operator(EP) as op:
                assert op.operator.shape == (14, 60, 14, 60)

            # Use the operator on the pdf
            evolved_pdfs, _integration_errors = apply_pdf(
                evolution_operator, pdf, x_grid
            )

        # Import the values of the LHA benchmark tables
        lha_benchmark = LHA_utils.compute_LHA_data(
            {"FNS": "FFNS", "PTO": 0, "XIF": 1}, {"polarized": False, "mugrid": 100.0}
        )

        # Make the test
        for flav in br.flavor_basis_pids:
            np.testing.assert_allclose(
                evolved_pdfs[EP][flav],
                (lha_benchmark["values"][EP[0]][flav] / x_grid),
                rtol=5e-4,
                atol=5e-6,
            )
