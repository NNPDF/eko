import io
import pathlib
import shutil
import tempfile

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
    def test_io(self, eko_factory: EKOFactory, tmp_path):
        # create object
        o1 = eko_factory.get()
        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        xgrid = eko_factory.operator.rotations.xgrid
        np.testing.assert_almost_equal(o1.xgrid.raw, xgrid.raw)
        np.testing.assert_almost_equal(o2.xgrid.raw, xgrid.raw)
        # fake eko.io files
        fpyaml = tmp_path / "test.yaml"
        legacy.dump_yaml_to_file(o1, fpyaml)
        # fake input file
        o3 = legacy.load_yaml_from_file(fpyaml)
        np.testing.assert_almost_equal(o3.xgrid.raw, xgrid.raw)
        # repeat for tar
        fptar = tmp_path / "test.tar"
        legacy.dump_tar(o1, fptar)
        o4 = legacy.load_tar(fptar)
        np.testing.assert_almost_equal(o4.xgrid.raw, xgrid.raw)
        fn = "test"
        with pytest.raises(ValueError, match="wrong suffix"):
            legacy.dump_tar(o1, fn)

    def test_rename_issue81(self):
        # https://github.com/N3PDF/eko/issues/81
        # create object
        o1, fake_card = fake_legacy
        for q2, op in fake_card["Q2grid"].items():
            o1[q2] = eko.io.Operator.from_dict(op)

        with tempfile.TemporaryDirectory() as folder:
            # dump
            p = pathlib.Path(folder)
            fp1 = p / "test1.tar"
            fp2 = p / "test2.tar"
            legacy.dump_tar(o1, fp1)
            # rename
            shutil.move(fp1, fp2)
            # reload
            o4 = legacy.load_tar(fp2)
            np.testing.assert_almost_equal(
                o4.xgrid.raw, fake_card["interpolation_xgrid"]
            )

    def test_io_bin(self):
        # create object
        o1, fake_card = fake_legacy
        for q2, op in fake_card["Q2grid"].items():
            o1[q2] = eko.io.Operator.from_dict(op)
        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream, False)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        np.testing.assert_almost_equal(o1.xgrid.raw, fake_card["interpolation_xgrid"])
        np.testing.assert_almost_equal(o2.xgrid.raw, fake_card["interpolation_xgrid"])


class TestManipulate:
    def test_xgrid_reshape(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        # create object
        mu2out = 10.0
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        eko_factory.operator.mu2grid = [mu2out]
        eko_factory.operator.rotations.xgrid = xg
        o1 = eko_factory.get()
        o1[mu2out] = eko.io.Operator(
            operator=eko_identity([1, 2, len(xg), 2, len(xg)])[0]
        )
        xgp = interpolation.XGrid(np.geomspace(1e-5, 1.0, 11))
        # only target
        otpath = tmp_path / "ot.tar"
        o1.deepcopy(otpath)
        with EKO.read(otpath) as ot:
            manipulate.xgrid_reshape(ot, xgp)
            chk_keys(o1.raw, ot.raw)
            assert ot[10].operator.shape == (2, len(xgp), 2, len(xg))
        ottpath = tmp_path / "ott.tar"
        o1.deepcopy(ottpath)
        with EKO.read(ottpath) as ott, pytest.warns(Warning):
            manipulate.xgrid_reshape(ott, xg)
            chk_keys(o1.raw, ott.raw)
            np.testing.assert_allclose(ott[10].operator, o1[10].operator)

        # only input
        oipath = tmp_path / "oi.tar"
        o1.deepcopy(oipath)
        with EKO.read(oipath) as oi:
            manipulate.xgrid_reshape(oi, inputgrid=xgp)
            assert oi[10].operator.shape == (2, len(xg), 2, len(xgp))
            chk_keys(o1.raw, oi.raw)
        oiipath = tmp_path / "oii.tar"
        o1.deepcopy(oiipath)
        with EKO.read(oiipath) as oii, pytest.warns(Warning):
            manipulate.xgrid_reshape(oii, inputgrid=xg)
            chk_keys(o1.raw, oii.raw)
            np.testing.assert_allclose(oii[10].operator, o1[10].operator)

        # both
        oitpath = tmp_path / "oit.tar"
        o1.deepcopy(oitpath)
        with EKO.read(oitpath) as oit:
            manipulate.xgrid_reshape(oit, xgp, xgp)
            chk_keys(o1.raw, oit.raw)
            op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
            np.testing.assert_allclose(oit[10].operator, op[0], atol=1e-10)

        # error
        with pytest.raises(ValueError):
            manipulate.xgrid_reshape(o1)

    def test_reshape_io(self, eko_factory: EKOFactory, tmp_path):
        eko_factory.path = tmp_path / "eko.tar"
        eko_factory.operator.rotations.pids = np.array([0, 1])
        eko_factory.operator.configs.interpolation_polynomial_degree = 1
        # create object
        o1 = eko_factory.get()
        path_copy = tmp_path / "eko_copy.tar"
        o1.deepcopy(path_copy)
        with EKO.edit(path_copy) as o2:
            manipulate.xgrid_reshape(
                o2, interpolation.XGrid([0.1, 1.0]), interpolation.XGrid([0.1, 1.0])
            )
            manipulate.flavor_reshape(o2, inputpids=np.array([[1, -1], [1, 1]]))
        # reload
        with EKO.read(path_copy) as o3:
            chk_keys(o1.raw, o3.raw)

    def test_flavor_reshape(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        # create object
        xg = interpolation.XGrid(np.geomspace(1e-5, 1.0, 21))
        pids = np.array([0, 1])
        mu2out = 10.0
        eko_factory.operator.rotations.xgrid = xg
        eko_factory.operator.rotations.pids = pids
        eko_factory.operator.mu2grid = np.array([mu2out])
        o1 = eko_factory.get()
        lpids = len(pids)
        lx = len(xg)
        o1[mu2out] = eko.io.Operator(
            operator=eko_identity([1, lpids, lx, lpids, lx])[0],
            error=None,
        )
        # only target
        target_r = np.array([[1, -1], [1, 1]])
        tpath = tmp_path / "ot.tar"
        ttpath = tmp_path / "ott.tar"
        o1.deepcopy(tpath)
        with EKO.edit(tpath) as ot:
            manipulate.flavor_reshape(ot, target_r)
            chk_keys(o1.raw, ot.raw)
            assert ot[mu2out].operator.shape == (2, len(xg), 2, len(xg))
            ot.deepcopy(ttpath)
        with EKO.edit(ttpath) as ott:
            manipulate.flavor_reshape(ott, np.linalg.inv(target_r))
            np.testing.assert_allclose(ott[mu2out].operator, o1[mu2out].operator)
            with pytest.warns(Warning):
                manipulate.flavor_reshape(ott, np.eye(2))
                chk_keys(o1.raw, ott.raw)
                np.testing.assert_allclose(ott[mu2out].operator, o1[mu2out].operator)

        # only input
        input_r = np.array([[1, -1], [1, 1]])
        ipath = tmp_path / "oi.tar"
        iipath = tmp_path / "oii.tar"
        o1.deepcopy(ipath)
        with EKO.edit(ipath) as oi:
            manipulate.flavor_reshape(oi, inputpids=input_r)
            chk_keys(o1.raw, oi.raw)
            assert oi[mu2out].operator.shape == (2, len(xg), 2, len(xg))
            oi.deepcopy(iipath)
        with EKO.edit(iipath) as oii:
            manipulate.flavor_reshape(oii, inputpids=np.linalg.inv(input_r))
            np.testing.assert_allclose(oii[mu2out].operator, o1[mu2out].operator)
            with pytest.warns(Warning):
                manipulate.flavor_reshape(oii, inputpids=np.eye(2))
                chk_keys(o1.raw, oii.raw)
                np.testing.assert_allclose(oii[mu2out].operator, o1[mu2out].operator)

        # both
        itpath = tmp_path / "oit.tar"
        o1.deepcopy(itpath)
        with EKO.edit(itpath) as oit:
            manipulate.flavor_reshape(
                oit, np.array([[1, -1], [1, 1]]), np.array([[1, -1], [1, 1]])
            )
            chk_keys(o1.raw, oit.raw)
            op = eko_identity([1, 2, len(xg), 2, len(xg)]).copy()
            np.testing.assert_allclose(oit[mu2out].operator, op[0], atol=1e-10)
        # error
        fpath = tmp_path / "fail.tar"
        o1.deepcopy(fpath)
        with pytest.raises(ValueError):
            with EKO.edit(fpath) as of:
                manipulate.flavor_reshape(of)

    def test_to_evol(self, eko_factory: EKOFactory, tmp_path):
        xgrid = interpolation.XGrid([0.5, 1.0])
        mu2_out = 2.0
        eko_factory.operator.mu0 = float(np.sqrt(1.0))
        eko_factory.operator.mu2grid = np.array([mu2_out])
        eko_factory.operator.rotations.xgrid = xgrid
        eko_factory.operator.rotations.pids = np.array(br.flavor_basis_pids)
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

        with (
            EKO.read(o01_path) as o01,
            EKO.read(o10_path) as o10,
            EKO.read(o11_path) as o11,
        ):
            # check the input rotated one
            np.testing.assert_allclose(
                o01.rotations.inputpids, br.rotate_flavor_to_evolution
            )
            np.testing.assert_allclose(o01.rotations.targetpids, br.flavor_basis_pids)
            # rotate also target
            manipulate.to_evol(o01, False, True)
            np.testing.assert_allclose(o01[mu2_out].operator, o11[mu2_out].operator)
            chk_keys(o00.raw, o01.raw)
            # check the target rotated one
            np.testing.assert_allclose(o10.rotations.inputpids, br.flavor_basis_pids)
            np.testing.assert_allclose(
                o10.rotations.targetpids, br.rotate_flavor_to_evolution
            )
            # rotate also input
            manipulate.to_evol(o10)
            np.testing.assert_allclose(o10[mu2_out].operator, o11[mu2_out].operator)
            chk_keys(o00.raw, o10.raw)
