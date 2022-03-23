# -*- coding: utf-8 -*-
import math

import numpy as np

from ekobox import gen_op as g_o
from ekobox import gen_theory as g_t
from ekobox import info_file


def test_build():
    op = g_o.gen_op_card([10, 100])
    theory = g_t.gen_theory_card(1, 10.0, update={"alphas": 0.2})
    info = info_file.build(
        theory, op, 4, info_update={"SetDesc": "Prova", "NewArg": 15.3, "MTop": 1.0}
    )
    assert info["AlphaS_MZ"] == 0.2
    assert info["SetDesc"] == "Prova"
    assert info["NewArg"] == 15.3
    assert info["NumMembers"] == 4
    assert info["MTop"] == theory["mt"]
    np.testing.assert_allclose(info["QMin"], math.sqrt(op["Q2grid"][0]), rtol=1e-5)
    assert info["XMin"] == op["xgrid"][0]
    assert info["XMax"] == op["xgrid"][-1] == 1.0
