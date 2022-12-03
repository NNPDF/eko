import math

import numpy as np

from eko import compatibility
from ekobox import cards, info_file


def test_build():
    op = cards.generate_operator([10, 100])
    theory = cards.generate_theory(1, 10.0, update={"alphas": 0.2})
    nt, no = compatibility.update(theory, op)
    info = info_file.build(
        nt, no, 4, info_update={"SetDesc": "Prova", "NewArg": 15.3, "MTop": 1.0}
    )
    assert info["AlphaS_MZ"] == 0.2
    assert info["SetDesc"] == "Prova"
    assert info["NewArg"] == 15.3
    assert info["NumMembers"] == 4
    assert info["MTop"] == nt["mt"]
    np.testing.assert_allclose(info["QMin"], math.sqrt(no["Q2grid"][0]), rtol=1e-5)
    assert info["XMin"] == no["rotations"]["xgrid"][0]
    assert info["XMax"] == no["rotations"]["xgrid"][-1] == 1.0
