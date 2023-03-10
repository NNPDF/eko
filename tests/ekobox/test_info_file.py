import math

import numpy as np

from ekobox import cards, info_file


def test_build():
    theory = cards.example.theory()
    theory.order = (2, 0)
    theory.couplings.alphas.value = 0.2
    op = cards.example.operator()
    op.mu0 = 1.0
    op.mugrid = np.array([10.0, 100.0])
    info = info_file.build(
        theory, op, 4, info_update={"SetDesc": "Prova", "NewArg": 15.3, "MTop": 1.0}
    )
    assert info["AlphaS_MZ"] == 0.2
    assert info["SetDesc"] == "Prova"
    assert info["NewArg"] == 15.3
    assert info["NumMembers"] == 4
    assert info["MTop"] == theory.quark_masses.t.value
    np.testing.assert_allclose(info["QMin"], math.sqrt(op.mu2grid[0]), rtol=1e-5)
    assert info["XMin"] == op.xgrid.raw[0]
    assert info["XMax"] == op.xgrid.raw[-1] == 1.0
