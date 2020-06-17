# -*- coding: utf-8 -*-

import numpy as np

import eko

def test_runner():
    """we don't check the content here, but only the shape"""
    setup = {
        "interpolation_xgrid": [.1, 1.],
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "alphas": 0.35,
        "Qref": np.sqrt(2),
        "Q0": np.sqrt(2),
        "FNS": "FFNS",
        "NfFF": 3,
        "Q2grid": [10,100]
    }
    r = eko.runner.Runner(setup)
    o = r.get_output()

    # check output = input
    np.testing.assert_allclose(o["interpolation_xgrid"],setup["interpolation_xgrid"])
    for k in ["interpolation_polynomial_degree", "interpolation_is_log"]:
        assert o[k] == setup[k]
    np.testing.assert_allclose(o["q2_ref"],setup["Q0"]**2)
    # check available operators
    assert len(o["Q2grid"]) == len(setup["Q2grid"])
    assert list(o["Q2grid"].keys()) == setup["Q2grid"]
    for ops in o["Q2grid"].values():
        assert "operators" in ops
        assert "operator_errors" in ops
        names = ["V.V","V3.V3","V8.V8","T3.T3","T8.T8","S.S","S.g","g.S","g.g"]
        assert list(ops["operators"].keys()) == names
        assert list(ops["operator_errors"].keys()) == names
