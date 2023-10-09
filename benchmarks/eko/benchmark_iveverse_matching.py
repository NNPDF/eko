import pathlib

import numpy as np
import pytest
from banana import toy

from eko import EKO
from eko.io import runcards
from eko.io.types import ReferenceRunning
from eko.runner.managed import solve
from ekobox import apply

here = pathlib.Path(__file__).parent.absolute()
MC = 1.51
C_PID = 4

# theory settings
th_raw = dict(
    order=[3, 0],
    couplings=dict(
        alphas=0.118,
        alphaem=0.007496252,
        scale=91.2,
        num_flavs_ref=5,
        max_num_flavs=6,
    ),
    heavy=dict(
        num_flavs_init=4,
        num_flavs_max_pdf=6,
        intrinsic_flavors=[],
        masses=[ReferenceRunning([mq, np.nan]) for mq in (MC, 4.92, 172.5)],
        masses_scheme="POLE",
        matching_ratios=[1.0, 1.0, np.inf],
    ),
    xif=1.0,
    n3lo_ad_variation=(0, 0, 0, 0, 0, 0, 0),
    matching_order=[2, 0],
)

# operator settings
op_raw = dict(
    mu0=1.65,
    xgrid=[0.0001, 0.001, 0.01, 0.1, 1],
    mugrid=[(MC, 3), (MC, 4)],
    configs=dict(
        evolution_method="truncated",
        ev_op_max_order=[1, 0],
        ev_op_iterations=1,
        interpolation_polynomial_degree=4,
        interpolation_is_log=True,
        scvar_method="exponentiated",
        inversion_method="exact",
        n_integration_cores=0,
        polarized=False,
        time_like=False,
    ),
    debug=dict(
        skip_singlet=False,
        skip_non_singlet=False,
    ),
)


@pytest.mark.isolated
def BenchmarkInverseMatching():
    th_card = runcards.TheoryCard.from_dict(th_raw)
    op_card = runcards.OperatorCard.from_dict(op_raw)

    eko_path2 = here / "test2.tar"
    eko_path2.unlink(missing_ok=True)
    solve(th_card, op_card, eko_path2)

    th_card.matching_order = [1, 0]
    eko_path1 = here / "test1.tar"
    eko_path1.unlink(missing_ok=True)
    solve(th_card, op_card, eko_path1)

    eko_output1 = EKO.read(eko_path1)
    eko_output2 = EKO.read(eko_path2)
    op1_nf3 = eko_output1[(MC**2, 3)]
    op2_nf3 = eko_output2[(MC**2, 3)]
    op1_nf4 = eko_output1[(MC**2, 4)]
    op2_nf4 = eko_output2[(MC**2, 4)]

    # test that nf=4 operators are the same
    np.testing.assert_allclose(op1_nf4.operator, op2_nf4.operator)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(op2_nf3.operator, op2_nf4.operator)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(op1_nf3.operator, op1_nf4.operator)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(op1_nf3.operator, op2_nf3.operator)

    pdf1 = apply.apply_pdf(eko_output1, toy.mkPDF("ToyLH", 0))
    pdf2 = apply.apply_pdf(eko_output2, toy.mkPDF("ToyLH", 0))

    # test that different PTO matching is applied correctly
    np.testing.assert_allclose(
        pdf1[(MC**2, 4)]["pdfs"][C_PID], pdf2[(MC**2, 4)]["pdfs"][C_PID]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            pdf1[(MC**2, 3)]["pdfs"][C_PID], pdf2[(MC**2, 3)]["pdfs"][C_PID]
        )
