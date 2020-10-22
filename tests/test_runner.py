# -*- coding: utf-8 -*-

import numpy as np

import eko


def test_runner():
    """we don't check the content here, but only the shape"""
    theory_card = {
        "alphas": 0.35,
        "PTO": 0,
        "XIF": 1.0,
        "XIR": 1.0,
        "Qref": np.sqrt(2),
        "Q0": np.sqrt(2),
        "FNS": "FFNS",
        "NfFF": 3,
        "ModEv": "EXA",
    }
    operators_card = {
        "Q2grid": [10, 100],
        "interpolation_xgrid": [0.1, 1.0],
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "debug_skip_singlet": True,
        "debug_skip_non_singlet": True,
        "ev_op_max_order": 1,
        "ev_op_iterations": 1,
    }
    r = eko.runner.Runner(theory_card, operators_card)
    o = r.get_output()

    # check output = input
    np.testing.assert_allclose(
        o["interpolation_xgrid"], operators_card["interpolation_xgrid"]
    )
    for k in ["interpolation_polynomial_degree", "interpolation_is_log"]:
        assert o[k] == operators_card[k]
    np.testing.assert_allclose(o["q2_ref"], theory_card["Q0"] ** 2)
    # check available operators
    assert len(o["Q2grid"]) == len(operators_card["Q2grid"])
    assert list(o["Q2grid"].keys()) == operators_card["Q2grid"]
    for ops in o["Q2grid"].values():
        assert "operators" in ops
        assert "operator_errors" in ops
        names = ["V.V", "V3.V3", "V8.V8", "T3.T3", "T8.T8", "S.S", "S.g", "g.S", "g.g"]
        assert list(ops["operators"].keys()) == names
        assert list(ops["operator_errors"].keys()) == names
