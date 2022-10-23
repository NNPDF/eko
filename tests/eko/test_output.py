# -*- coding: utf-8 -*-
import copy
import io
import pathlib
import shutil
import tempfile

import numpy as np
import pytest

from eko import basis_rotation as br
from eko import interpolation, output
from eko.output import legacy, manipulate, struct
from ekobox.mock import eko_identity


def chk_keys(a, b):
    """Check all keys are preserved"""
    assert sorted(a.keys()) == sorted(b.keys())
    for key, value in a.items():
        if isinstance(value, dict):
            assert sorted(value.keys()) == sorted(b[key].keys())


# class TestLegacy:
#     def test_io(self, fake_legacy, tmp_path):
#         # create object
#         o1, fake_card = fake_legacy
#         for q2, op in fake_card["Q2grid"].items():
#             o1[q2] = output.Operator.from_dict(op)

#         # test streams
#         stream = io.StringIO()
#         legacy.dump_yaml(o1, stream)
#         # rewind and read again
#         stream.seek(0)
#         o2 = legacy.load_yaml(stream)
#         np.testing.assert_almost_equal(o1.xgrid.raw, fake_card["interpolation_xgrid"])
#         np.testing.assert_almost_equal(o2.xgrid.raw, fake_card["interpolation_xgrid"])
#         # fake output files
#         fpyaml = tmp_path / "test.yaml"
#         legacy.dump_yaml_to_file(o1, fpyaml)
#         # fake input file
#         o3 = legacy.load_yaml_from_file(fpyaml)
#         np.testing.assert_almost_equal(o3.xgrid.raw, fake_card["interpolation_xgrid"])
#         # repeat for tar
#         fptar = tmp_path / "test.tar"
#         legacy.dump_tar(o1, fptar)
#         o4 = legacy.load_tar(fptar)
#         np.testing.assert_almost_equal(o4.xgrid.raw, fake_card["interpolation_xgrid"])
#         fn = "test"
#         with pytest.raises(ValueError, match="wrong suffix"):
#             legacy.dump_tar(o1, fn)

#     def test_rename_issue81(self, fake_legacy):
#         # https://github.com/N3PDF/eko/issues/81
#         # create object
#         o1, fake_card = fake_legacy
#         for q2, op in fake_card["Q2grid"].items():
#             o1[q2] = output.Operator.from_dict(op)

#         with tempfile.TemporaryDirectory() as folder:
#             # dump
#             p = pathlib.Path(folder)
#             fp1 = p / "test1.tar"
#             fp2 = p / "test2.tar"
#             legacy.dump_tar(o1, fp1)
#             # rename
#             shutil.move(fp1, fp2)
#             # reload
#             o4 = legacy.load_tar(fp2)
#             np.testing.assert_almost_equal(
#                 o4.xgrid.raw, fake_card["interpolation_xgrid"]
#             )

#     def test_io_bin(self, fake_legacy):
#         # create object
#         o1, fake_card = fake_legacy
#         for q2, op in fake_card["Q2grid"].items():
#             o1[q2] = output.Operator.from_dict(op)
#         # test streams
#         stream = io.StringIO()
#         legacy.dump_yaml(o1, stream, False)
#         # rewind and read again
#         stream.seek(0)
#         o2 = legacy.load_yaml(stream)
#         np.testing.assert_almost_equal(o1.xgrid.raw, fake_card["interpolation_xgrid"])
#         np.testing.assert_almost_equal(o2.xgrid.raw, fake_card["interpolation_xgrid"])


class TestManipulate:
    def test_xgrid_reshape(self, tmp_path, default_cards):
        # create object
        xg = np.geomspace(1e-5, 1.0, 21).tolist()
        tc, oc = default_cards
        oc["xgrid"] = xg
        oc["rotations"]["targetgrid"] = xg
        oc["rotations"]["inputgrid"] = xg
        q2, op = 10.0, output.Operator.from_dict(
            dict(
                operator=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                error=np.zeros((2, len(xg), 2, len(xg))),
            )
        )
        xgp = interpolation.XGrid(np.geomspace(1e-5, 1.0, 11))
        with struct.EKO.create(tc, oc) as eko0:
            eko0[q2] = op

            # only target
            pt = tmp_path / "ekot.tar"
            ptt = tmp_path / "ekott.tar"
            eko0.deepcopy(pt)
            with struct.EKO.open(pt) as ekot:
                manipulate.xgrid_reshape(ekot, xgp)
                with ekot.operator(q2) as op:
                    assert op.operator.shape == (2, len(xgp), 2, len(xg))
                ekot.deepcopy(ptt)
            # revert operation should return to original
            with struct.EKO.open(ptt) as ekott:
                with pytest.warns(Warning):
                    manipulate.xgrid_reshape(ekott, interpolation.XGrid(xg))
                with ekott.operator(q2) as op:
                    np.testing.assert_allclose(ekott[10].operator, op.operator)

            # only input
            pi = tmp_path / "ekoi.tar"
            pii = tmp_path / "ekoii.tar"
            eko0.deepcopy(pi)
            with struct.EKO.open(pi) as ekoi:
                manipulate.xgrid_reshape(ekoi, inputgrid=xgp)
                with ekoi.operator(q2) as op:
                    assert op.operator.shape == (2, len(xg), 2, len(xgp))
                ekoi.deepcopy(pii)
            # revert operation should return to original
            with struct.EKO.open(pii) as ekoii:
                with pytest.warns(Warning):
                    manipulate.xgrid_reshape(ekoii, interpolation.XGrid(xg))
                with ekoii.operator(q2) as op:
                    np.testing.assert_allclose(ekoii[10].operator, op.operator)

            # both
            pit = tmp_path / "ekoit.tar"
            eko0.deepcopy(pit)
            with struct.EKO.open(pit) as ekoit:
                manipulate.xgrid_reshape(ekoit, xgp, xgp)
                with ekoit.operator(q2) as op:
                    assert op.operator.shape == (2, len(xgp), 2, len(xgp))
                    id_op = eko_identity([1, 2, len(xgp), 2, len(xgp)])
                    np.testing.assert_allclose(op.operator, id_op[0], atol=1e-10)
            # error
            with pytest.raises(ValueError):
                manipulate.xgrid_reshape(eko0, {})

    # def test_reshape_io(self, fake_output):
    #     # create object
    #     o1, fake_card = fake_output
    #     for q2, op in fake_card["Q2grid"].items():
    #         o1[q2] = output.Operator.from_dict(op)
    #     o2 = copy.deepcopy(o1)
    #     manipulate.xgrid_reshape(
    #         o2, interpolation.XGrid([0.1, 1.0]), interpolation.XGrid([0.1, 1.0])
    #     )
    #     manipulate.flavor_reshape(o2, inputpids=np.array([[1, -1], [1, 1]]))
    #     # dump
    #     stream = io.StringIO()
    #     legacy.dump_yaml(o2, stream)
    #     # reload
    #     stream.seek(0)
    #     o3 = legacy.load_yaml(stream)
    #     chk_keys(o1.raw, o3.raw)

    def test_flavor_reshape(self, tmp_path, default_cards):
        # create object
        xg = np.geomspace(1e-5, 1.0, 21).tolist()
        tc, oc = default_cards
        oc["xgrid"] = xg
        oc["rotations"]["inputpids"] = [1, 2]
        oc["rotations"]["targetpids"] = [1, 2]
        q2, op = 10.0, output.Operator.from_dict(
            dict(
                operator=eko_identity([1, 2, len(xg), 2, len(xg)])[0],
                error=np.zeros((2, len(xg), 2, len(xg))),
            )
        )
        with struct.EKO.create(tc, oc) as eko0:
            eko0[q2] = op

            # only target
            pt = tmp_path / "ekot.tar"
            ptt = tmp_path / "ekott.tar"
            eko0.deepcopy(pt)
            with struct.EKO.open(pt) as ekot:
                target_r = np.array([[1, -1], [1, 1]])
                manipulate.flavor_reshape(ekot, target_r)
                with ekot.operator(q2) as op:
                    assert op.operator.shape == (2, len(xg), 2, len(xg))
                ekot.deepcopy(ptt)
            # revert operation should return to original
            with struct.EKO.open(ptt) as ekott:
                with pytest.warns(Warning):
                    manipulate.flavor_reshape(ekott, np.linalg.inv(target_r))
                with ekott.operator(q2) as op:
                    np.testing.assert_allclose(ekott[10].operator, op.operator)

            # only input
            pi = tmp_path / "ekoi.tar"
            pii = tmp_path / "ekoii.tar"
            eko0.deepcopy(pi)
            with struct.EKO.open(pi) as ekoi:
                input_r = np.array([[1, -1], [1, 1]])
                manipulate.flavor_reshape(ekoi, inputpids=input_r)
                with ekoi.operator(q2) as op:
                    assert op.operator.shape == (2, len(xg), 2, len(xg))
                ekoi.deepcopy(pii)
            # revert operation should return to original
            with struct.EKO.open(pii) as ekoii:
                with pytest.warns(Warning):
                    manipulate.flavor_reshape(ekoii, np.linalg.inv(input_r))
                with ekoii.operator(q2) as op:
                    np.testing.assert_allclose(ekoii[10].operator, op.operator)

            # both
            pit = tmp_path / "ekoit.tar"
            eko0.deepcopy(pit)
            with struct.EKO.open(pit) as ekoit:
                manipulate.flavor_reshape(
                    ekoit, np.array([[1, -1], [1, 1]]), np.array([[1, -1], [1, 1]])
                )
                with ekoit.operator(q2) as op:
                    assert op.operator.shape == (2, len(xg), 2, len(xg))
                    id_op = eko_identity([1, 2, len(xg), 2, len(xg)])
                    np.testing.assert_allclose(op.operator, id_op[0], atol=1e-10)
            # error
            with pytest.raises(TypeError):
                manipulate.flavor_reshape(eko0, {})

    def test_to_evol(self, fake_factory, tmp_path):
        xgrid = np.array([0.5, 1.0])
        interpolation_polynomial_degree = 1
        interpolation_is_log = False
        q2_ref = 1
        q2_out = 2
        Q2grid = fake_factory.mk_g([q2_out], len(br.flavor_basis_pids), len(xgrid))
        d = dict(
            Q0=np.sqrt(q2_ref),
            Q2grid=Q2grid,
            rotations=dict(
                xgrid=xgrid,
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
        o00 = output.EKO.new(theory={}, operator=d)
        o00[q2_out] = output.Operator(**Q2grid[q2_out])
        o01 = o00.deepcopy(tmp_path / "o01.tar")
        manipulate.to_evol(o01)
        o10 = o00.deepcopy(tmp_path / "o10.tar")
        manipulate.to_evol(o10, False, True)
        o11 = o00.deepcopy(tmp_path / "o11.tar")
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
