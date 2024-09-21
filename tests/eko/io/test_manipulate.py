import numpy as np
import pytest

import eko
from eko import basis_rotation as br
from eko import interpolation
from eko.io import manipulate
from ekobox.mock import eko_identity


def chk_keys(a, b):
    """Check all keys are preserved."""
    assert sorted(a.keys()) == sorted(b.keys())
    for key, value in a.items():
        if isinstance(value, dict):
            assert sorted(value.keys()) == sorted(b[key].keys())


class TestManipulate:
    def test_xgrid_reshape(self):
        # create object
        interpdeg = 1
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        xgp = interpolation.XGrid(np.geomspace(1e-5, 1.0, 11))
        lpids = 2
        o1 = eko.io.Operator(
            operator=eko_identity([1, lpids, len(xg), lpids, len(xg)])[0]
        )
        # only target
        ot = manipulate.xgrid_reshape(o1, xg, interpdeg, xgp)
        assert ot.operator.shape == (lpids, len(xgp), lpids, len(xg))
        ott = manipulate.xgrid_reshape(ot, xgp, interpdeg, xg)
        # when blowing up again a line 0 ... 0 0   1 0   0 ... 0 becomes
        #                              0 ... 0 0.5 0 0.5 0 ... 0 instead
        np.testing.assert_allclose(
            np.sum(ott.operator, axis=3), np.sum(o1.operator, axis=3)
        )

        # only input
        oi = manipulate.xgrid_reshape(o1, xg, interpdeg, inputgrid=xgp)
        assert oi.operator.shape == (lpids, len(xg), lpids, len(xgp))
        oii = manipulate.xgrid_reshape(oi, xgp, interpdeg, inputgrid=xg)
        np.testing.assert_allclose(
            np.sum(oii.operator, axis=3), np.sum(o1.operator, axis=3)
        )
        with pytest.warns(Warning):
            oiii = manipulate.xgrid_reshape(oii, xg, interpdeg, inputgrid=xg)
            np.testing.assert_allclose(oiii.operator, oii.operator)

        # both
        oit = manipulate.xgrid_reshape(o1, xg, interpdeg, xgp, xgp)
        op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
        np.testing.assert_allclose(oit.operator, op[0], atol=1e-10)
        # op error handling
        o1e = eko.io.Operator(
            operator=eko_identity([1, lpids, len(xg), lpids, len(xg)])[0],
            error=0.1 * eko_identity([1, lpids, len(xg), lpids, len(xg)])[0],
        )
        assert ot.error is None
        assert oi.error is None
        ot2 = manipulate.xgrid_reshape(o1e, xg, interpdeg, xgp)
        assert ot2.error is not None

        # Python error
        with pytest.raises(ValueError, match="Nor inputgrid nor targetgrid"):
            manipulate.xgrid_reshape(o1, xg, interpdeg)

    def test_flavor_reshape(self):
        # create object
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        lpids = len(br.flavor_basis_pids)
        lx = len(xg)
        o1 = eko.io.Operator(
            operator=eko_identity([1, lpids, lx, lpids, lx])[0],
            error=None,
        )

        # only input
        input_r = np.eye(lpids)
        input_r[:2, :2] = np.array([[1, -1], [1, 1]])
        oi = manipulate.flavor_reshape(o1, inputpids=input_r)
        assert oi.operator.shape == (lpids, len(xg), lpids, len(xg))
        oii = manipulate.flavor_reshape(oi, inputpids=np.linalg.inv(input_r))
        np.testing.assert_allclose(oii.operator, o1.operator)
        with pytest.warns(Warning):
            oiii = manipulate.flavor_reshape(oii, inputpids=np.eye(lpids))
            np.testing.assert_allclose(oiii.operator, oii.operator)

        # only target
        target_r = np.eye(lpids)
        target_r[:2, :2] = np.array([[1, -1], [1, 1]])
        ot = manipulate.flavor_reshape(o1, target_r)
        assert ot.operator.shape == (lpids, len(xg), lpids, len(xg))
        ott = manipulate.flavor_reshape(ot, np.linalg.inv(target_r))
        np.testing.assert_allclose(ott.operator, o1.operator)
        with pytest.warns(Warning):
            ottt = manipulate.flavor_reshape(ott, np.eye(lpids))
            np.testing.assert_allclose(ottt.operator, ott.operator)

        # both
        oit = manipulate.flavor_reshape(o1, target_r, input_r)
        op = eko_identity([1, lpids, len(xg), lpids, len(xg)]).copy()
        np.testing.assert_allclose(oit.operator, op[0], atol=1e-10)
        # error
        with pytest.raises(ValueError, match="Nor inputpids nor targetpids"):
            manipulate.flavor_reshape(o1)

    def test_to_evol(self):
        self._test_to_all_evol(
            manipulate.to_evol,
            br.rotate_flavor_to_evolution,
        )

    def test_to_uni_evol(self):
        self._test_to_all_evol(
            manipulate.to_uni_evol,
            br.rotate_flavor_to_unified_evolution,
        )

    def _test_to_all_evol(self, to_evol_fnc, rot_matrix):
        # create object
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        lpids = len(br.flavor_basis_pids)
        lx = len(xg)
        o = eko.io.Operator(
            operator=eko_identity([1, lpids, lx, lpids, lx])[0],
            error=None,
        )

        # do it once
        o01 = to_evol_fnc(o, True, False)
        o10 = to_evol_fnc(o, False, True)
        o11 = to_evol_fnc(o, True, True)

        # do also the other one
        np.testing.assert_allclose(
            to_evol_fnc(o01, False, True).operator, o11.operator, atol=1e-15
        )
        np.testing.assert_allclose(
            to_evol_fnc(o10, True, False).operator, o11.operator, atol=1e-15
        )
