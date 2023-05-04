import numpy as np

from eko import basis_rotation as br
from eko import interpolation
from eko.io.bases import Bases


class TestBases:
    def test_fallback(self):
        xg = interpolation.XGrid([0.1, 1.0])
        r = Bases(xgrid=xg)
        np.testing.assert_allclose(r.targetpids, r.pids)
        np.testing.assert_allclose(r.inputpids, r.pids)
        assert r.xgrid == xg
        assert r.targetgrid == xg
        assert r.inputgrid == xg

    def test_overwrite(self):
        tpids = np.array([3, 4] + list(br.flavor_basis_pids[2:]))
        ipids = np.array([5, 6] + list(br.flavor_basis_pids[2:]))
        xg = interpolation.XGrid([0.1, 1.0])
        txg = interpolation.XGrid([0.2, 1.0])
        ixg = interpolation.XGrid([0.3, 1.0])
        r = Bases(
            xgrid=xg,
            _targetgrid=txg,
            _inputgrid=ixg,
            _targetpids=tpids,
            _inputpids=ipids,
        )
        np.testing.assert_allclose(r.targetpids, tpids)
        np.testing.assert_allclose(r.inputpids, ipids)
        assert r.xgrid == xg
        assert r.targetgrid == txg
        assert r.inputgrid == ixg

    def test_init(self):
        xg = interpolation.XGrid([0.1, 1.0])
        txg = np.array([0.2, 1.0])
        ixg = {"grid": [0.3, 1.0], "log": True}
        r = Bases(xgrid=xg, _targetgrid=txg, _inputgrid=ixg)
        assert isinstance(r.xgrid, interpolation.XGrid)
        assert isinstance(r.targetgrid, interpolation.XGrid)
        assert isinstance(r.inputgrid, interpolation.XGrid)
        assert r.xgrid == xg
        assert r.targetgrid == interpolation.XGrid(txg)
        assert r.inputgrid == interpolation.XGrid.load(ixg)
