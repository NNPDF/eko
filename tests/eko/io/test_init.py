import pathlib

import numpy as np
import pytest

import eko
from eko import EKO
from eko import basis_rotation as br
from eko import interpolation
from eko.io import legacy, manipulate, runcards
from ekobox.mock import eko_identity
from tests.conftest import EKOFactory


def chk_keys(a, b):
    """Check all keys are preserved"""
    assert sorted(a.keys()) == sorted(b.keys())
    for key, value in a.items():
        if isinstance(value, dict):
            assert sorted(value.keys()) == sorted(b[key].keys())


class TestLegacy:
    def test_load_tar(self, out_v0, tmp_path: pathlib.Path):
        oppath = tmp_path / "eko.tar"
        with pytest.warns(UserWarning, match="alphas"):
            legacy.load_tar(out_v0, oppath)
        with EKO.read(oppath) as eko:
            assert eko.metadata.data_version == 0


class TestManipulate:
    def test_xgrid_reshape(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        # create object
        muout = 10.0
        mu2out = muout**2
        epout = (mu2out, 5)
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        eko_factory.operator.mugrid = [(muout, 5)]
        eko_factory.operator.xgrid = xg
        o1 = eko_factory.get()
        lpids = 2
        o1[epout] = eko.io.Operator(
            operator=eko_identity([1, lpids, len(xg), lpids, len(xg)])[0]
        )
        xgp = interpolation.XGrid(np.geomspace(1e-5, 1.0, 11))
        # only target
        otpath = tmp_path / "ot.tar"
        o1.deepcopy(otpath)
        with EKO.edit(otpath) as ot:
            manipulate.xgrid_reshape(ot, xgp)
            chk_keys(o1.raw, ot.raw)
            assert ot[epout].operator.shape == (lpids, len(xgp), lpids, len(xg))
        ottpath = tmp_path / "ott.tar"
        o1.deepcopy(ottpath)
        with EKO.edit(ottpath) as ott:
            with pytest.warns(Warning):
                manipulate.xgrid_reshape(ott, xg)
                chk_keys(o1.raw, ott.raw)
                np.testing.assert_allclose(ott[epout].operator, o1[epout].operator)

        # only input
        oipath = tmp_path / "oi.tar"
        o1.deepcopy(oipath)
        with EKO.edit(oipath) as oi:
            manipulate.xgrid_reshape(oi, inputgrid=xgp)
            assert oi[epout].operator.shape == (lpids, len(xg), lpids, len(xgp))
            chk_keys(o1.raw, oi.raw)
        oiipath = tmp_path / "oii.tar"
        o1.deepcopy(oiipath)
        with EKO.edit(oiipath) as oii:
            with pytest.warns(Warning):
                manipulate.xgrid_reshape(oii, inputgrid=xg)
                chk_keys(o1.raw, oii.raw)
                np.testing.assert_allclose(oii[epout].operator, o1[epout].operator)

        # both
        oitpath = tmp_path / "oit.tar"
        o1.deepcopy(oitpath)
        with EKO.edit(oitpath) as oit:
            manipulate.xgrid_reshape(oit, xgp, xgp)
            chk_keys(o1.raw, oit.raw)
            op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
            np.testing.assert_allclose(oit[epout].operator, op[0], atol=1e-10)

        # error
        with pytest.raises(ValueError):
            manipulate.xgrid_reshape(o1)

    def test_reshape_io(self, eko_factory: EKOFactory, tmp_path):
        eko_factory.path = tmp_path / "eko.tar"
        eko_factory.operator.configs.interpolation_polynomial_degree = 1
        # create object
        o1 = eko_factory.get()
        lpids = len(o1.rotations.pids)
        path_copy = tmp_path / "eko_copy.tar"
        o1.deepcopy(path_copy)
        newxgrid = interpolation.XGrid([0.1, 1.0])
        inputpids = np.eye(lpids)
        inputpids[:2, :2] = np.array([[1, -1], [1, 1]])
        with EKO.edit(path_copy) as o2:
            manipulate.xgrid_reshape(o2, newxgrid, newxgrid)
            manipulate.flavor_reshape(o2, inputpids=inputpids)
        # reload
        with EKO.read(path_copy) as o3:
            chk_keys(o1.raw, o3.raw)
            np.testing.assert_allclose(o3.rotations.inputgrid.raw, newxgrid.raw)
            np.testing.assert_allclose(o3.rotations.targetgrid.raw, newxgrid.raw)
            # since we use a general rotation, the inputpids are erased,
            # leaving just as many zeros as PIDs, as placeholders for missing
            # values
            np.testing.assert_allclose(
                o3.rotations.inputpids, [0] * len(o3.rotations.pids)
            )
            # these has to be unchanged
            np.testing.assert_allclose(o3.rotations.targetpids, o3.rotations.pids)

    def test_flavor_reshape(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        # create object
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        muout = 10.0
        mu2out = muout**2
        eko_factory.operator.xgrid = xg
        eko_factory.operator.mugrid = [(muout, 5)]
        o1 = eko_factory.get()
        lpids = len(o1.rotations.pids)
        lx = len(xg)
        o1[mu2out] = eko.io.Operator(
            operator=eko_identity([1, lpids, lx, lpids, lx])[0],
            error=None,
        )
        # only target
        target_r = np.eye(lpids)
        target_r[:2, :2] = np.array([[1, -1], [1, 1]])
        tpath = tmp_path / "ot.tar"
        ttpath = tmp_path / "ott.tar"
        o1.deepcopy(tpath)
        with EKO.edit(tpath) as ot:
            manipulate.flavor_reshape(ot, target_r)
            chk_keys(o1.raw, ot.raw)
            assert ot[mu2out].operator.shape == (lpids, len(xg), lpids, len(xg))
            ot.deepcopy(ttpath)
        with EKO.edit(ttpath) as ott:
            manipulate.flavor_reshape(ott, np.linalg.inv(target_r))
            np.testing.assert_allclose(ott[mu2out].operator, o1[mu2out].operator)
            with pytest.warns(Warning):
                manipulate.flavor_reshape(ott, np.eye(lpids))
                chk_keys(o1.raw, ott.raw)
                np.testing.assert_allclose(ott[mu2out].operator, o1[mu2out].operator)

        # only input
        input_r = np.eye(lpids)
        input_r[:2, :2] = np.array([[1, -1], [1, 1]])
        ipath = tmp_path / "oi.tar"
        iipath = tmp_path / "oii.tar"
        o1.deepcopy(ipath)
        with EKO.edit(ipath) as oi:
            manipulate.flavor_reshape(oi, inputpids=input_r)
            chk_keys(o1.raw, oi.raw)
            assert oi[mu2out].operator.shape == (lpids, len(xg), lpids, len(xg))
            oi.deepcopy(iipath)
        with EKO.edit(iipath) as oii:
            manipulate.flavor_reshape(oii, inputpids=np.linalg.inv(input_r))
            np.testing.assert_allclose(oii[mu2out].operator, o1[mu2out].operator)
            with pytest.warns(Warning):
                manipulate.flavor_reshape(oii, inputpids=np.eye(lpids))
                chk_keys(o1.raw, oii.raw)
                np.testing.assert_allclose(oii[mu2out].operator, o1[mu2out].operator)

        # both
        itpath = tmp_path / "oit.tar"
        o1.deepcopy(itpath)
        with EKO.edit(itpath) as oit:
            manipulate.flavor_reshape(oit, target_r, input_r)
            chk_keys(o1.raw, oit.raw)
            op = eko_identity([1, lpids, len(xg), lpids, len(xg)]).copy()
            np.testing.assert_allclose(oit[mu2out].operator, op[0], atol=1e-10)
        # error
        fpath = tmp_path / "fail.tar"
        o1.deepcopy(fpath)
        with pytest.raises(ValueError):
            with EKO.edit(fpath) as of:
                manipulate.flavor_reshape(of)

    def test_to_evol(self, eko_factory: EKOFactory, tmp_path):
        xgrid = interpolation.XGrid([0.5, 1.0])
        mu_out = 2.0
        mu2_out = mu_out**2
        eko_factory.operator.mu0 = float(np.sqrt(1.0))
        eko_factory.operator.mugrid = [(mu_out, 4)]
        eko_factory.operator.xgrid = xgrid
        eko_factory.operator.configs.interpolation_polynomial_degree = 1
        eko_factory.operator.configs.interpolation_is_log = False
        eko_factory.operator.configs.ev_op_max_order = (2, 0)
        eko_factory.operator.configs.ev_op_iterations = 1
        eko_factory.operator.configs.inversion_method = runcards.InversionMethod.EXACT
        o00 = eko_factory.get()
        o01_path = tmp_path / "o01.tar"
        o00.deepcopy(o01_path)
        with EKO.edit(o01_path) as o01:
            manipulate.to_evol(o01)
        o10_path = tmp_path / "o10.tar"
        o00.deepcopy(o10_path)
        with EKO.edit(o10_path) as o10:
            manipulate.to_evol(o10, False, True)
        o11_path = tmp_path / "o11.tar"
        o00.deepcopy(o11_path)
        with EKO.edit(o11_path) as o11:
            manipulate.to_evol(o11, True, True)
            chk_keys(o00.raw, o11.raw)

        with EKO.edit(o01_path) as o01:
            with EKO.edit(o10_path) as o10:
                with EKO.read(o11_path) as o11:
                    # check the input rotated one
                    np.testing.assert_allclose(
                        o01.rotations.inputpids, br.rotate_flavor_to_evolution
                    )
                    np.testing.assert_allclose(
                        o01.rotations.targetpids, br.flavor_basis_pids
                    )
                    # rotate also target
                    manipulate.to_evol(o01, False, True)
                    np.testing.assert_allclose(
                        o01[mu2_out].operator, o11[mu2_out].operator
                    )
                    chk_keys(o00.raw, o01.raw)
                    # check the target rotated one
                    np.testing.assert_allclose(
                        o10.rotations.inputpids, br.flavor_basis_pids
                    )
                    np.testing.assert_allclose(
                        o10.rotations.targetpids, br.rotate_flavor_to_evolution
                    )
                    # rotate also input
                    manipulate.to_evol(o10)
                    np.testing.assert_allclose(
                        o10[mu2_out].operator, o11[mu2_out].operator
                    )
                    chk_keys(o00.raw, o10.raw)
