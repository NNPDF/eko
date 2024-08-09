import numpy as np

from eko import basis_rotation as br


def check_shapes(o, txs, ixs, theory_card, operators_card):
    tpids = len(br.flavor_basis_pids)
    ipids = len(br.flavor_basis_pids)
    op_shape = (tpids, len(txs), ipids, len(ixs))

    # check output = input
    np.testing.assert_allclose(o.xgrid.raw, operators_card.xgrid.raw)
    # targetgrid and inputgrid in the opcard are now ignored, we are testing this
    np.testing.assert_allclose(
        o.xgrid.raw,
        txs.raw,
    )
    np.testing.assert_allclose(o.xgrid.raw, ixs.raw)
    np.testing.assert_allclose(o.mu20, operators_card.mu20)
    # check available operators
    ~o.operators
    assert len(o.mu2grid) == len(operators_card.mu2grid)
    assert list(o.mu2grid) == operators_card.mu2grid
    for _, ops in o.items():
        assert ops.operator.shape == op_shape
        assert ops.error.shape == op_shape
