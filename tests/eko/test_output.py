# -*- coding: utf-8 -*-
import copy
import io
import pathlib
import shutil
import tempfile
from unittest import mock

import numpy as np
import pytest

from eko import basis_rotation as br
from eko import output
from eko.output import legacy, manipulate


def eko_identity(shape):
    i, k = np.ogrid[: shape[1], : shape[2]]
    eko_identity = np.zeros(shape[1:], int)
    eko_identity[i, k, i, k] = 1
    return np.broadcast_to(eko_identity[np.newaxis, :, :, :, :], shape)


def chk_keys(a, b):
    """Check all keys are preserved"""
    assert sorted(a.keys()) == sorted(b.keys())
    for key, value in a.items():
        if isinstance(value, dict):
            assert sorted(value.keys()) == sorted(b[key].keys())


class TestLegacy:
    def test_io(self, fake_output):
        # create object
        o1 = output.EKO.from_dict(fake_output)
        for q2, op in fake_output["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)
        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        np.testing.assert_almost_equal(o1.xgrid, fake_output["xgrid"])
        np.testing.assert_almost_equal(o2.xgrid, fake_output["xgrid"])
        # fake output files
        m_out = mock.mock_open(read_data="")
        with mock.patch("builtins.open", m_out) as mock_file:
            fn = "test.yaml"
            legacy.dump_yaml_to_file(o1, fn)
            mock_file.assert_called_with(fn, "w", encoding="utf-8")
        # fake input file
        stream.seek(0)
        m_in = mock.mock_open(read_data=stream.getvalue())
        with mock.patch("builtins.open", m_in) as mock_file:
            fn = "test.yaml"
            o3 = legacy.load_yaml_from_file(fn)
            mock_file.assert_called_with(fn, encoding="utf-8")
            np.testing.assert_almost_equal(o3.xgrid, fake_output["xgrid"])
        # repeat for tar
        fn = "test.tar"
        with tempfile.TemporaryDirectory() as folder:
            fp = pathlib.Path(folder) / fn
            legacy.dump_tar(o1, fp)
            o4 = legacy.load_tar(fp)
            np.testing.assert_almost_equal(o4.xgrid, fake_output["xgrid"])
        fn = "test"
        with pytest.raises(ValueError, match="wrong suffix"):
            legacy.dump_tar(o1, fn)

    def test_rename_issue81(self, fake_output):
        # https://github.com/N3PDF/eko/issues/81
        # create object
        o1 = output.EKO.from_dict(fake_output)
        for q2, op in fake_output["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)

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
            np.testing.assert_almost_equal(o4.xgrid, fake_output["xgrid"])

    def test_io_bin(self, fake_output):
        # create object
        o1 = output.EKO.from_dict(fake_output)
        for q2, op in fake_output["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)
        # test streams
        stream = io.StringIO()
        legacy.dump_yaml(o1, stream, False)
        # rewind and read again
        stream.seek(0)
        o2 = legacy.load_yaml(stream)
        np.testing.assert_almost_equal(o1.xgrid, fake_output["xgrid"])
        np.testing.assert_almost_equal(o2.xgrid, fake_output["xgrid"])


class TestManipulate:
    def test_xgrid_reshape(self, fake_output):
        # create object
        xg = np.geomspace(1e-5, 1.0, 21)
        o1 = output.EKO.from_dict(fake_output)
        o1.xgrid = xg
        o1.rotations.targetgrid = xg
        o1.rotations.inputgrid = xg
        o1._operators = {
            10: output.Operator.from_dict(
                dict(
                    operator=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                    error=np.zeros((2, len(xg), 2, len(xg))),
                    alphas=np.random.rand(),
                )
            )
        }
        xgp = np.geomspace(1e-5, 1.0, 11)
        # only target
        ot = copy.deepcopy(o1)
        manipulate.xgrid_reshape(ot, xgp, inplace=True)
        chk_keys(o1.raw, ot.raw)
        assert ot[10].operator.shape == (2, len(xgp), 2, len(xg))
        ott = copy.deepcopy(o1)
        with pytest.warns(Warning):
            manipulate.xgrid_reshape(ott, xg)
            chk_keys(o1.raw, ott.raw)
            np.testing.assert_allclose(ott[10].operator, o1[10].operator)

        # only input
        oi = copy.deepcopy(o1)
        manipulate.xgrid_reshape(oi, inputgrid=xgp)
        assert oi[10].operator.shape == (2, len(xg), 2, len(xgp))
        chk_keys(o1.raw, oi.raw)
        oii = copy.deepcopy(o1)
        with pytest.warns(Warning):
            manipulate.xgrid_reshape(oii, inputgrid=xg)
            chk_keys(o1.raw, oii.raw)
            np.testing.assert_allclose(oii[10].operator, o1[10].operator)

        # both
        oit = copy.deepcopy(o1)
        manipulate.xgrid_reshape(oit, xgp, xgp)
        chk_keys(o1.raw, oit.raw)
        op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
        np.testing.assert_allclose(oit[10].operator, op[0], atol=1e-10)
        # error
        with pytest.raises(ValueError):
            manipulate.xgrid_reshape(copy.deepcopy(o1))

    def test_reshape_io(self, fake_output):
        # create object
        o1 = output.EKO.from_dict(fake_output)
        for q2, op in fake_output["Q2grid"].items():
            o1[q2] = output.Operator.from_dict(op)
        o2 = copy.deepcopy(o1)
        manipulate.xgrid_reshape(o2, [0.1, 1.0], [0.1, 1.0])
        manipulate.flavor_reshape(o2, inputbasis=np.array([[1, -1], [1, 1]]))
        # dump
        stream = io.StringIO()
        legacy.dump_yaml(o2, stream)
        # reload
        stream.seek(0)
        o3 = legacy.load_yaml(stream)
        chk_keys(o1.raw, o3.raw)

    def test_flavor_reshape(self, fake_output):
        # create object
        xg = np.geomspace(1e-5, 1.0, 21)
        o1 = output.EKO.from_dict(fake_output)
        o1.xgrid = xg
        o1.rotations.targetgrid = xg
        o1.rotations.inputgrid = xg
        o1._operators = {
            10: output.Operator.from_dict(
                dict(
                    operator=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                    error=np.zeros((2, len(xg), 2, len(xg))),
                    alphas=np.random.rand(),
                )
            )
        }
        # only target
        target_r = np.array([[1, -1], [1, 1]])
        ot = copy.deepcopy(o1)
        manipulate.flavor_reshape(ot, target_r)
        chk_keys(o1.raw, ot.raw)
        assert ot[10].operator.shape == (2, len(xg), 2, len(xg))
        ott = copy.deepcopy(ot)
        manipulate.flavor_reshape(ott, np.linalg.inv(target_r))
        np.testing.assert_allclose(ott[10].operator, o1[10].operator)
        with pytest.warns(Warning):
            manipulate.flavor_reshape(ott, np.eye(2))
            chk_keys(o1.raw, ott.raw)
            np.testing.assert_allclose(ott[10].operator, o1[10].operator)

        # only input
        input_r = np.array([[1, -1], [1, 1]])
        oi = copy.deepcopy(o1)
        manipulate.flavor_reshape(oi, inputbasis=input_r)
        chk_keys(o1.raw, oi.raw)
        assert oi[10].operator.shape == (2, len(xg), 2, len(xg))
        oii = copy.deepcopy(oi)
        manipulate.flavor_reshape(oii, inputbasis=np.linalg.inv(input_r))
        np.testing.assert_allclose(oii[10].operator, o1[10].operator)
        with pytest.warns(Warning):
            manipulate.flavor_reshape(oii, inputbasis=np.eye(2))
            chk_keys(o1.raw, oii.raw)
            np.testing.assert_allclose(oii[10].operator, o1[10].operator)

        # both
        oit = copy.deepcopy(o1)
        manipulate.flavor_reshape(
            oit, np.array([[1, -1], [1, 1]]), np.array([[1, -1], [1, 1]])
        )
        chk_keys(o1.raw, oit.raw)
        op = eko_identity([1, 2, len(xg), 2, len(xg)]).copy()
        np.testing.assert_allclose(oit[10].operator, op[0], atol=1e-10)
        # error
        with pytest.raises(ValueError):
            manipulate.flavor_reshape(copy.deepcopy(o1))

    def test_to_evol(self, fake_factory):
        xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        q2_ref = 1
        q2_out = 2
        Q2grid = fake_factory.mk_g([q2_out], len(br.flavor_basis_pids), len(xgrid))
        d = dict(
            xgrid=xgrid,
            Q0=np.sqrt(q2_ref),
            Q2grid=Q2grid,
            rotations=dict(
                targetgrid=xgrid,
                inputgrid=xgrid,
                inputpids=br.flavor_basis_pids,
                targetpids=br.flavor_basis_pids,
            ),
            configs=dict(
                interpolation_polynomial_degree=interpolation_polynomial_degree,
                interpolation_is_log=interpolation_is_log,
                ev_op_max_order=1,
                ev_op_iterations=1,
                backward_inversion="exact",
            ),
        )
        o00 = output.EKO.from_dict(d)
        o00[q2_out] = Q2grid[q2_out]
        o01 = copy.deepcopy(o00)
        manipulate.to_evol(o01)
        o10 = copy.deepcopy(o00)
        manipulate.to_evol(o10, False, True)
        o11 = copy.deepcopy(o00)
        manipulate.to_evol(o11, True, True)
        chk_keys(o00.raw, o11.raw)

        # check the input rotated one
        np.testing.assert_allclose(o01.rotations.inputpids, br.evol_basis_pids)
        np.testing.assert_allclose(o01.rotations.targetpids, br.flavor_basis_pids)
        # rotate also target
        manipulate.to_evol(o01, False, True)
        np.testing.assert_allclose(o01[q2_out].operator, o11[q2_out].operator)
        chk_keys(o00.raw, o01.raw)
        # check the target rotated one
        np.testing.assert_allclose(o10.rotations.inputpids, br.flavor_basis_pids)
        np.testing.assert_allclose(o10.rotations.targetpids, br.evol_basis_pids)
        # rotate also input
        manipulate.to_evol(o10)
        np.testing.assert_allclose(o10[q2_out].operator, o11[q2_out].operator)
        chk_keys(o00.raw, o10.raw)
