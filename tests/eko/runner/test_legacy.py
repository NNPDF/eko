import copy

import numpy as np

import eko
from eko import basis_rotation as br
from eko.interpolation import XGrid


def test_raw(theory_card, operator_card, tmp_path):
    """we don't check the content here, but only the shape"""
    tc = theory_card
    oc = operator_card
    r = eko.runner.legacy.Runner(tc, oc, path=tmp_path / "eko.tar")
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)


def check_shapes(o, txs, ixs, theory_card, operators_card):
    tpids = len(o.rotations.targetpids)
    ipids = len(o.rotations.inputpids)
    op_shape = (tpids, len(txs), ipids, len(ixs))

    op_targetgrid = operators_card.rotations.targetgrid
    op_inputgrid = operators_card.rotations.inputgrid
    # check output = input
    np.testing.assert_allclose(o.xgrid.raw, operators_card.rotations.xgrid.raw)
    # targetgrid and inputgrid in the opcard are now ignored, we are testing this
    np.testing.assert_allclose(
        o.rotations.targetgrid.raw,
        txs.raw,
    )
    np.testing.assert_allclose(o.rotations.inputgrid.raw, ixs.raw)
    np.testing.assert_allclose(o.mu20, operators_card.mu20)
    # check available operators
    assert len(o.mu2grid) == len(operators_card.mu2grid)
    assert list(o.mu2grid) == operators_card.mu2grid
    for _, ops in o.items():
        assert ops.operator.shape == op_shape
        assert ops.error.shape == op_shape


def test_vfns(theory_ffns, operator_card, tmp_path):
    # change targetpids
    tc = theory_ffns(3)
    oc = copy.deepcopy(operator_card)
    tc.matching.c = 1.0
    tc.matching.b = 1.0
    tc.order = (2, 0)
    oc.debug.skip_non_singlet = False
    r = eko.runner.legacy.Runner(tc, oc, path=tmp_path / "eko.tar")
    o = r.get_output()
    check_shapes(o, o.xgrid, o.xgrid, tc, oc)
