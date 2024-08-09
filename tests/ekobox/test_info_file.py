import math

import numpy as np

from ekobox import cards, info_file


def test_build():
    theory = cards.example.theory()
    theory.order = (2, 0)
    theory.couplings.alphas = 0.2
    op = cards.example.operator()
    op.init = (1.0, 3)
    op.mugrid = [(10.0, 5), (100.0, 5)]
    info = info_file.build(
        theory, op, 4, info_update={"SetDesc": "Prova", "NewArg": 15.3, "MTop": 1.0}
    )
    assert info["AlphaS_MZ"] == 0.2
    assert info["SetDesc"] == "Prova"
    assert info["NewArg"] == 15.3
    assert info["NumMembers"] == 4
    assert info["MTop"] == theory.heavy.masses.t.value
    np.testing.assert_allclose(info["QMin"], math.sqrt(op.mu2grid[0]), rtol=1e-5)
    assert info["XMin"] == op.xgrid.raw[0]
    assert info["XMax"] == op.xgrid.raw[-1] == 1.0


def test_build_alphas_good():
    """Good configurations."""
    theory = cards.example.theory()
    theory.order = (2, 0)
    theory.couplings.alphas = 0.2
    op = cards.example.operator()
    op.mu0 = 1.0
    # base case
    op.mugrid = [(100.0, 5), (10.0, 5)]
    info = info_file.build_alphas(theory, op)
    assert len(info["AlphaS_Vals"]) == 2
    np.testing.assert_allclose(info["AlphaS_Qs"], [10.0, 100.0])
    # several nf
    op.mugrid = [(5.0, 4), (10.0, 5), (1.0, 3), (5.0, 3), (10.0, 5), (100.0, 5)]
    info = info_file.build_alphas(theory, op)
    assert len(info["AlphaS_Vals"]) == 6
    np.testing.assert_allclose(info["AlphaS_Qs"], [1.0, 5.0, 5.0, 10.0, 10.0, 100.0])
    # several nf with gap
    op.mugrid = [(1.0, 3), (10.0, 3), (10.0, 5), (100.0, 5)]
    info = info_file.build_alphas(theory, op)
    assert len(info["AlphaS_Vals"]) == 4
    np.testing.assert_allclose(info["AlphaS_Qs"], [1.0, 10.0, 10.0, 100.0])
