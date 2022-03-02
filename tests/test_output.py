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
from ekomark import apply


class FakePDF:
    def hasFlavor(self, pid):
        return pid == 1

    def xfxQ2(self, _pid, x, _q2):
        return x


def eko_identity(shape):
    i, k = np.ogrid[: shape[1], : shape[2]]
    eko_identity = np.zeros(shape[1:], int)
    eko_identity[i, k, i, k] = 1
    return np.broadcast_to(eko_identity[np.newaxis, :, :, :, :], shape)


def chk_keys(a, b):
    """Check all keys are preserved"""
    assert sorted(a.keys()) == sorted(b.keys())
    for q2, op in a["Q2grid"].items():
        assert q2 in b["Q2grid"]
        opb = b["Q2grid"][q2]
        assert sorted(op.keys()) == sorted(opb.keys())
        assert op["alphas"] == opb["alphas"]


class TestOutput:
    shape = (2, 2)

    def mkO(self):
        ma, mae = np.random.rand(2, *self.shape)
        return ma, mae

    def mk_g(self, q2s, lpids, lx):
        Q2grid = {}
        for q2 in q2s:
            Q2grid[q2] = {
                "operators": np.random.rand(lpids, lx, lpids, lx),
                "operator_errors": np.random.rand(lpids, lx, lpids, lx),
                "alphas": np.random.rand(),
            }
        return Q2grid

    def fake_output(self):
        # build data
        interpolation_xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        pids = [0, 1]
        q2_ref = 1
        q2_out = 2
        Q2grid = self.mk_g([q2_out], len(pids), len(interpolation_xgrid))
        d = dict(
            interpolation_xgrid=interpolation_xgrid,
            targetgrid=interpolation_xgrid,
            inputgrid=interpolation_xgrid,
            interpolation_polynomial_degree=interpolation_polynomial_degree,
            interpolation_is_log=interpolation_is_log,
            q2_ref=q2_ref,
            inputpids=pids,
            targetpids=pids,
            Q2grid=Q2grid,
        )
        return d

    def test_io(self):
        d = self.fake_output()
        # create object
        o1 = output.Output(d)
        # test streams
        stream = io.StringIO()
        o1.dump_yaml(stream)
        # rewind and read again
        stream.seek(0)
        o2 = output.Output.load_yaml(stream)
        np.testing.assert_almost_equal(
            o1["interpolation_xgrid"], d["interpolation_xgrid"]
        )
        np.testing.assert_almost_equal(
            o2["interpolation_xgrid"], d["interpolation_xgrid"]
        )
        # fake output files
        m_out = mock.mock_open(read_data="")
        with mock.patch("builtins.open", m_out) as mock_file:
            fn = "test.yaml"
            o1.dump_yaml_to_file(fn)
            mock_file.assert_called_with(fn, "w", encoding="utf-8")
        # fake input file
        stream.seek(0)
        m_in = mock.mock_open(read_data=stream.getvalue())
        with mock.patch("builtins.open", m_in) as mock_file:
            fn = "test.yaml"
            o3 = output.Output.load_yaml_from_file(fn)
            mock_file.assert_called_with(fn, encoding="utf-8")
            np.testing.assert_almost_equal(
                o3["interpolation_xgrid"], d["interpolation_xgrid"]
            )
        # repeat for tar
        fn = "test.tar"
        with tempfile.TemporaryDirectory() as folder:
            fp = pathlib.Path(folder) / fn
            o1.dump_tar(fp)
            o4 = output.Output.load_tar(fp)
            np.testing.assert_almost_equal(
                o4["interpolation_xgrid"], d["interpolation_xgrid"]
            )
        fn = "test"
        with pytest.raises(ValueError, match="wrong suffix"):
            o1.dump_tar(fn)

    def test_rename_issue81(self):
        # https://github.com/N3PDF/eko/issues/81
        d = self.fake_output()
        # create object
        o1 = output.Output(d)

        with tempfile.TemporaryDirectory() as folder:
            # dump
            p = pathlib.Path(folder)
            fp1 = p / "test1.tar"
            fp2 = p / "test2.tar"
            o1.dump_tar(fp1)
            # rename
            shutil.move(fp1, fp2)
            # reload
            o4 = output.Output.load_tar(fp2)
            np.testing.assert_almost_equal(
                o4["interpolation_xgrid"], d["interpolation_xgrid"]
            )

    def test_io_bin(self):
        d = self.fake_output()
        # create object
        o1 = output.Output(d)
        # test streams
        stream = io.StringIO()
        o1.dump_yaml(stream, False)
        # rewind and read again
        stream.seek(0)
        o2 = output.Output.load_yaml(stream)
        np.testing.assert_almost_equal(
            o1["interpolation_xgrid"], d["interpolation_xgrid"]
        )
        np.testing.assert_almost_equal(
            o2["interpolation_xgrid"], d["interpolation_xgrid"]
        )

    def test_apply(self):
        d = self.fake_output()
        q2_out = list(d["Q2grid"].keys())[0]
        # create object
        o = output.Output(d)
        # fake pdfs
        pdf = FakePDF()
        pdf_grid = apply.apply_pdf(o, pdf)
        assert len(pdf_grid) == len(d["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["targetpids"]
        ref_pid1 = d["Q2grid"][q2_out]["operators"][0, :, 1, :] @ np.ones(
            len(d["interpolation_xgrid"])
        )
        np.testing.assert_allclose(pdfs[0], ref_pid1)
        ref_pid2 = d["Q2grid"][q2_out]["operators"][1, :, 1, :] @ np.ones(
            len(d["interpolation_xgrid"])
        )
        np.testing.assert_allclose(pdfs[1], ref_pid2)
        # rotate to target_grid
        target_grid = [0.75]
        pdf_grid = apply.apply_pdf(o, pdf, target_grid)
        assert len(pdf_grid) == 1
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == d["targetpids"]

    def test_apply_flavor(self, monkeypatch):
        d = self.fake_output()
        q2_out = list(d["Q2grid"].keys())[0]
        # create object
        o = output.Output(d)
        # fake pdfs
        pdf = FakePDF()
        monkeypatch.setattr(
            "eko.basis_rotation.rotate_flavor_to_evolution", np.ones((2, 2))
        )
        monkeypatch.setattr("eko.basis_rotation.flavor_basis_pids", d["targetpids"])
        fake_evol_basis = ("a", "b")
        monkeypatch.setattr("eko.basis_rotation.evol_basis", fake_evol_basis)
        pdf_grid = apply.apply_pdf(o, pdf, rotate_to_evolution_basis=True)
        assert len(pdf_grid) == len(d["Q2grid"])
        pdfs = pdf_grid[q2_out]["pdfs"]
        assert list(pdfs.keys()) == list(fake_evol_basis)
        ref_a = (
            d["Q2grid"][q2_out]["operators"][0, :, 1, :]
            + d["Q2grid"][q2_out]["operators"][1, :, 1, :]
        ) @ np.ones(len(d["interpolation_xgrid"]))
        np.testing.assert_allclose(pdfs["a"], ref_a)

    def test_xgrid_reshape(self):
        d = self.fake_output()
        # create object
        xg = np.geomspace(1e-5, 1.0, 21)
        o1 = output.Output(d)
        o1["interpolation_xgrid"] = xg
        o1["targetgrid"] = xg
        o1["inputgrid"] = xg
        o1["Q2grid"] = {
            10: dict(
                operators=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                operator_errors=np.zeros((2, len(xg), 2, len(xg))),
                alphas=np.random.rand(),
            )
        }
        xgp = np.geomspace(1e-5, 1.0, 11)
        # only target
        ot = copy.deepcopy(o1)
        ot.xgrid_reshape(xgp)
        chk_keys(o1, ot)
        assert ot["Q2grid"][10]["operators"].shape == (2, len(xgp), 2, len(xg))
        ott = copy.deepcopy(o1)
        with pytest.warns(Warning):
            ott.xgrid_reshape(xg)
            chk_keys(o1, ott)
            np.testing.assert_allclose(
                ott["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
            )

        # only input
        oi = copy.deepcopy(o1)
        oi.xgrid_reshape(inputgrid=xgp)
        assert oi["Q2grid"][10]["operators"].shape == (2, len(xg), 2, len(xgp))
        chk_keys(o1, oi)
        oii = copy.deepcopy(o1)
        with pytest.warns(Warning):
            oii.xgrid_reshape(inputgrid=xg)
            chk_keys(o1, oii)
            np.testing.assert_allclose(
                oii["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
            )

        # both
        oit = copy.deepcopy(o1)
        oit.xgrid_reshape(xgp, xgp)
        chk_keys(o1, oit)
        op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
        np.testing.assert_allclose(oit["Q2grid"][10]["operators"], op[0], atol=1e-10)
        # error
        with pytest.raises(ValueError):
            copy.deepcopy(o1).xgrid_reshape()

    def test_reshape_io(self):
        d = self.fake_output()
        # create object
        o1 = output.Output(d)
        o2 = copy.deepcopy(o1)
        o2.xgrid_reshape([0.1, 1.0], [0.1, 1.0])
        o2.flavor_reshape(inputbasis=np.array([[1, -1], [1, 1]]))
        # dump
        stream = io.StringIO()
        o2.dump_yaml(stream)
        # reload
        stream.seek(0)
        o3 = output.Output.load_yaml(stream)
        # eko_version is only added in get_raw
        del o3["eko_version"]
        chk_keys(o1, o3)

    def test_flavor_reshape(self):
        d = self.fake_output()
        # create object
        xg = np.geomspace(1e-5, 1.0, 21)
        o1 = output.Output(d)
        o1["interpolation_xgrid"] = xg
        o1["targetgrid"] = xg
        o1["inputgrid"] = xg
        o1["Q2grid"] = {
            10: dict(
                operators=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                operator_errors=np.zeros((2, len(xg), 2, len(xg))),
                alphas=np.random.rand(),
            )
        }
        # only target
        target_r = np.array([[1, -1], [1, 1]])
        ot = copy.deepcopy(o1)
        ot.flavor_reshape(target_r)
        chk_keys(o1, ot)
        assert ot["Q2grid"][10]["operators"].shape == (2, len(xg), 2, len(xg))
        ott = copy.deepcopy(ot)
        ott.flavor_reshape(np.linalg.inv(target_r))
        np.testing.assert_allclose(
            ott["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
        )
        with pytest.warns(Warning):
            ott.flavor_reshape(np.eye(2))
            chk_keys(o1, ott)
            np.testing.assert_allclose(
                ott["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
            )

        # only input
        input_r = np.array([[1, -1], [1, 1]])
        oi = copy.deepcopy(o1)
        oi.flavor_reshape(inputbasis=input_r)
        chk_keys(o1, oi)
        assert oi["Q2grid"][10]["operators"].shape == (2, len(xg), 2, len(xg))
        oii = copy.deepcopy(oi)
        oii.flavor_reshape(inputbasis=np.linalg.inv(input_r))
        np.testing.assert_allclose(
            oii["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
        )
        with pytest.warns(Warning):
            oii.flavor_reshape(inputbasis=np.eye(2))
            chk_keys(o1, oii)
            np.testing.assert_allclose(
                oii["Q2grid"][10]["operators"], o1["Q2grid"][10]["operators"]
            )

        # both
        oit = copy.deepcopy(o1)
        oit.flavor_reshape(np.array([[1, -1], [1, 1]]), np.array([[1, -1], [1, 1]]))
        chk_keys(o1, oit)
        op = eko_identity([1, 2, len(xg), 2, len(xg)]).copy()
        np.testing.assert_allclose(oit["Q2grid"][10]["operators"], op[0], atol=1e-10)
        # error
        with pytest.raises(ValueError):
            copy.deepcopy(o1).flavor_reshape()

    def test_to_evol(self):
        interpolation_xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        q2_ref = 1
        q2_out = 2
        Q2grid = self.mk_g(
            [q2_out], len(br.flavor_basis_pids), len(interpolation_xgrid)
        )
        d = dict(
            interpolation_xgrid=interpolation_xgrid,
            targetgrid=interpolation_xgrid,
            inputgrid=interpolation_xgrid,
            interpolation_polynomial_degree=interpolation_polynomial_degree,
            interpolation_is_log=interpolation_is_log,
            q2_ref=q2_ref,
            inputpids=br.flavor_basis_pids,
            targetpids=br.flavor_basis_pids,
            Q2grid=Q2grid,
        )
        o00 = output.Output(d)
        o01 = copy.copy(o00)
        o01.to_evol()
        o10 = copy.copy(o00)
        o10.to_evol(False, True)
        o11 = copy.copy(o00)
        o11.to_evol(True, True)
        chk_keys(o00, o11)

        # check the input rotated one
        np.testing.assert_allclose(o01["inputpids"], br.evol_basis_pids)
        np.testing.assert_allclose(o01["targetpids"], br.flavor_basis_pids)
        # rotate also target
        o01.to_evol(False, True)
        np.testing.assert_allclose(
            o01["Q2grid"][q2_out]["operators"], o11["Q2grid"][q2_out]["operators"]
        )
        chk_keys(o00, o01)
        # check the target rotated one
        np.testing.assert_allclose(o10["inputpids"], br.flavor_basis_pids)
        np.testing.assert_allclose(o10["targetpids"], br.evol_basis_pids)
        # rotate also input
        o10.to_evol()
        np.testing.assert_allclose(
            o10["Q2grid"][q2_out]["operators"], o11["Q2grid"][q2_out]["operators"]
        )
        chk_keys(o00, o10)
