# -*- coding: utf-8 -*-

import numpy as np

import eko


def test_runner():
    """we don't check the content here, but only the shape"""
    theory_card = {
        "alphas": 0.35,
        "PTO": 0,
        "fact_to_ren_scale_ratio": 1.0,
        "Qref": np.sqrt(2),
        "Q0": np.sqrt(2),
        "FNS": "FFNS",
        "NfFF": 3,
        "ModEv": "EXA",
        "IC": 0,
        "mc": 1.0,
        "mb": 4.75,
        "mt": 173.0,
        "kcThr": 0,
        "kbThr": np.inf,
        "ktThr": np.inf,
        "MaxNfPdf": 6,
        "MaxNfAs": 6,
    }
    operators_card = {
        "Q2grid": [10, 100],
        "interpolation_xgrid": [0.01, 0.1, 1.0],
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "debug_skip_singlet": True,
        "debug_skip_non_singlet": True,
        "ev_op_max_order": 1,
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
    }
    r = eko.runner.Runner(theory_card, operators_card)
    o = r.get_output()
    # lx = len()
    check_shapes(
        o,
        o["interpolation_xgrid"],
        o["interpolation_xgrid"],
        theory_card,
        operators_card,
    )
    # change targetgrid
    tgrid = [0.1, 1.0]
    operators_card["targetgrid"] = tgrid
    rr = eko.runner.Runner(theory_card, operators_card)
    oo = rr.get_output()
    check_shapes(
        oo,
        tgrid,
        o["interpolation_xgrid"],
        theory_card,
        operators_card,
    )


def check_shapes(o, txs, ixs, theory_card, operators_card):
    lpids = len(o["pids"])
    op_shape = (lpids, len(txs), lpids, len(ixs))

    # check output = input
    np.testing.assert_allclose(
        o["interpolation_xgrid"], operators_card["interpolation_xgrid"]
    )
    np.testing.assert_allclose(o["targetgrid"], txs)
    np.testing.assert_allclose(o["inputgrid"], ixs)
    for k in ["interpolation_polynomial_degree", "interpolation_is_log"]:
        assert o[k] == operators_card[k]
    np.testing.assert_allclose(o["q2_ref"], theory_card["Q0"] ** 2)
    # check available operators
    assert len(o["Q2grid"]) == len(operators_card["Q2grid"])
    assert list(o["Q2grid"].keys()) == operators_card["Q2grid"]
    for ops in o["Q2grid"].values():
        assert "operators" in ops
        assert "operator_errors" in ops
        assert ops["operators"].shape == op_shape
        assert ops["operator_errors"].shape == op_shape
