# -*- coding: utf-8 -*-
import copy

import numpy as np

import eko
import eko.interpolation
from eko import basis_rotation as br

theory_card = {
    "alphas": 0.35,
    "alphaqed": 0.007496,
    "PTO": 0,
    "QED": 0,
    "fact_to_ren_scale_ratio": 1.0,
    "Qref": np.sqrt(2),
    "nfref": 4,
    "Q0": np.sqrt(2),
    "nf0": 4,
    "FNS": "FFNS",
    "NfFF": 3,
    "ModEv": "EXA",
    "IC": 0,
    "IB": 0,
    "mc": 1.0,
    "mb": 4.75,
    "mt": 173.0,
    "kcThr": 0,
    "kbThr": np.inf,
    "ktThr": np.inf,
    "MaxNfPdf": 6,
    "MaxNfAs": 6,
    "HQ": "MSBAR",
    "Qmc": 1.0,
    "Qmb": 4.75,
    "Qmt": 173.0,
    "ModSV": None,
}
operators_card = {
    "Q2grid": [10, 100],
    "configs": {
        "interpolation_polynomial_degree": 1,
        "interpolation_is_log": True,
        "ev_op_max_order": (2, 0),
        "ev_op_iterations": 1,
        "backward_inversion": "exact",
        "n_integration_cores": 1,
    },
    "rotations": {
        "xgrid": [0.01, 0.1, 1.0],
        "pids": np.array(br.flavor_basis_pids),
        "inputgrid": None,
        "targetgrid": None,
        "inputpids": None,
        "targetpids": None,
    },
    "debug": {"skip_singlet": True, "skip_non_singlet": True},
}


def test_raw():
    """we don't check the content here, but only the shape"""
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    r = eko.runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)


def test_targetgrid():
    # change targetgrid
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    tgrid = [0.1, 1.0]
    oc["rotations"]["targetgrid"] = tgrid
    r = eko.runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, eko.interpolation.XGrid(np.array(tgrid)), o.xgrid, tc, oc)
    # check actual value
    np.testing.assert_allclose(o.rotations.targetgrid, tgrid)


def test_rotate_pids():
    # change pids
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    oc["rotations"]["targetpids"] = np.eye(14) + 0.1 * np.random.rand(14, 14)
    oc["rotations"]["inputpids"] = np.eye(14) + 0.1 * np.random.rand(14, 14)
    r = eko.runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)
    # check actual values
    assert o.rotations.targetpids == [0] * 14
    assert o.rotations.inputpids == [0] * 14


def check_shapes(o, txs, ixs, theory_card, operators_card):
    tpids = len(o.rotations.targetpids)
    ipids = len(o.rotations.inputpids)
    op_shape = (tpids, len(txs), ipids, len(ixs))

    # check output = input
    np.testing.assert_allclose(o.xgrid.raw, operators_card["rotations"]["xgrid"])
    np.testing.assert_allclose(o.rotations.targetgrid.raw, txs.raw)
    np.testing.assert_allclose(o.rotations.inputgrid.raw, ixs.raw)
    for k in ["interpolation_polynomial_degree", "interpolation_is_log"]:
        assert getattr(o.configs, k) == operators_card["configs"][k]
    np.testing.assert_allclose(o.Q02, theory_card["Q0"] ** 2)
    # check available operators
    assert len(o.Q2grid) == len(operators_card["Q2grid"])
    assert list(o.Q2grid) == operators_card["Q2grid"]
    for _, ops in o.items():
        assert ops.operator.shape == op_shape
        assert ops.error.shape == op_shape


def test_vfns():
    # change targetpids
    tc = copy.deepcopy(theory_card)
    oc = copy.deepcopy(operators_card)
    tc["kcThr"] = 1.0
    tc["kbThr"] = 1.0
    tc["order"] = (3, 0)
    oc["debug"]["skip_non_singlet"] = False
    # tc,oc = compatibility.update(theory_card,operators_card)
    r = eko.runner.Runner(tc, oc)
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)
