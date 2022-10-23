# -*- coding: utf-8 -*-
import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from eko import compatibility, interpolation, output
from eko.output import struct
from ekobox import operators_card as oc
from ekobox import theory_card as tc


@dataclass
class MyDictLike(struct.DictLike):
    l: npt.NDArray
    f: float
    x: interpolation.XGrid
    t: tuple
    s: str


def test_DictLike():
    d = MyDictLike.from_dict(
        dict(
            l=np.arange(5.0),
            f=np.arange(5.0)[-1],
            x=interpolation.XGrid([0.1, 1.0]),
            t=(1.0, 2.0),
            s="s",
        )
    )
    assert d.f == 4.0
    dd = MyDictLike.from_dict(d.raw)
    assert dd.f == 4.0
    # check we can dump and reload
    stream = io.StringIO()
    yaml.safe_dump(d.raw, stream)
    stream.seek(0)
    ddd = yaml.safe_load(stream)
    assert "l" in ddd
    np.testing.assert_allclose(ddd["l"], np.arange(5.0))


class TestOperator:
    def test_value_only(self):
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        assert opv.error is None
        for compress in (True, False):
            stream = io.BytesIO()
            opv.save(stream, compress)
            stream.seek(0)
            opv_ = struct.Operator.load(stream, compress)
            np.testing.assert_allclose(opv.operator, opv_.operator)
            np.testing.assert_allclose(v, opv_.operator)
            assert opv_.error is None

    def test_value_and_error(self):
        v, e = np.random.rand(2, 2, 2)
        opve = struct.Operator(operator=v, error=e)
        for compress in (True, False):
            stream = io.BytesIO()
            opve.save(stream, compress)
            stream.seek(0)
            opve_ = struct.Operator.load(stream, compress)
            np.testing.assert_allclose(opve.operator, opve_.operator)
            np.testing.assert_allclose(v, opve_.operator)
            np.testing.assert_allclose(opve.error, opve_.error)
            np.testing.assert_allclose(e, opve_.error)


class TestRotations:
    def test_fallback(self):
        pids = np.array([1, 2])
        xg = interpolation.XGrid([0.1, 1.0])
        r = struct.Rotations(xgrid=xg, pids=pids)
        np.testing.assert_allclose(r.pids, pids)
        np.testing.assert_allclose(r.targetpids, pids)
        np.testing.assert_allclose(r.inputpids, pids)
        assert r.xgrid == xg
        assert r.targetgrid == xg
        assert r.inputgrid == xg

    def test_overwrite(self):
        pids = np.array([1, 2])
        tpids = np.array([3, 4])
        ipids = np.array([5, 6])
        xg = interpolation.XGrid([0.1, 1.0])
        txg = interpolation.XGrid([0.2, 1.0])
        ixg = interpolation.XGrid([0.3, 1.0])
        r = struct.Rotations(
            xgrid=xg,
            pids=pids,
            _targetgrid=txg,
            _inputgrid=ixg,
            _targetpids=tpids,
            _inputpids=ipids,
        )
        np.testing.assert_allclose(r.pids, pids)
        np.testing.assert_allclose(r.targetpids, tpids)
        np.testing.assert_allclose(r.inputpids, ipids)
        assert r.xgrid == xg
        assert r.targetgrid == txg
        assert r.inputgrid == ixg

    def test_init(self):
        pids = np.array([1, 2])
        xg = interpolation.XGrid([0.1, 1.0])
        txg = np.array([0.2, 1.0])
        ixg = {"grid": [0.3, 1.0], "log": True}
        r = struct.Rotations(xgrid=xg, pids=pids, _targetgrid=txg, _inputgrid=ixg)
        assert isinstance(r.xgrid, interpolation.XGrid)
        assert isinstance(r.targetgrid, interpolation.XGrid)
        assert isinstance(r.inputgrid, interpolation.XGrid)
        assert r.xgrid == xg
        assert r.targetgrid == interpolation.XGrid(txg)
        assert r.inputgrid == interpolation.XGrid.load(ixg)


class TestEKO:
    def _default_cards(self):
        t = tc.generate(0, 1.0)
        o = oc.generate([10.0])
        return compatibility.update(t, o)

    def test_new_error(self, tmp_path):
        nt, no = self._default_cards()
        # try to write to a file different from bla
        no_tar_path = tmp_path / "Blub.bla"
        with pytest.raises(struct.OutputNotTar):
            struct.EKO.new(nt, no, no_tar_path)
        # try to overwrite an existing file
        exists_path = tmp_path / "Blub.tar"
        exists_path.write_text("Blub", encoding="utf-8")
        with pytest.raises(struct.OutputExistsError):
            struct.EKO.new(nt, no, exists_path)

    def test_load_error(self, tmp_path):
        # try to read from a non-tar path
        no_tar_path = tmp_path / "Blub.tar"
        no_tar_path.write_text("Blub", encoding="utf-8")
        with pytest.raises(struct.OutputNotTar):
            struct.EKO.open_tar(no_tar_path)

    def test_properties(self):
        with struct.EKO.create(*self._default_cards()) as ekoo:
            assert "mc" in ekoo.theory_card
            assert "debug" in ekoo.operator_card
            np.testing.assert_allclose(ekoo.Q2grid, np.array([10.0]))
            assert 10.0 in ekoo
            default_grid = interpolation.XGrid(ekoo.operator_card["rotations"]["xgrid"])
            assert ekoo.xgrid == default_grid
            for use_target in (True, False):
                assert ekoo.interpolator(False, use_target).xgrid == default_grid
            xg = interpolation.XGrid([0.1, 1.0])
            ekoo.xgrid = xg
            assert ekoo.xgrid == xg
            assert "debug" in ekoo.raw
            # # check we can dump and reload
            # stream = io.StringIO()
            # yaml.safe_dump(ekoo.raw, stream)
            # stream.seek(0)
            # raw_eko = yaml.safe_load(stream)
            # assert "debug" in raw_eko

    def test_ops(self):
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        with struct.EKO.create(*self._default_cards()) as ekoo:
            # try setting not an operator
            with pytest.raises(ValueError):
                ekoo[10.0] = "bla"
            # approx
            ekoo[10.0] = opv
            assert ekoo.approx(20.0) is None
            assert ekoo.approx(11.0, atol=2) == 10.0
            ekoo[11.0] = opv
            with pytest.raises(ValueError):
                ekoo.approx(10.5, atol=2)
            # iterate
            for q2, q2eko in zip((10.0, 11.0), ekoo):
                assert q2 == q2eko
                np.testing.assert_allclose(v, ekoo[q2].operator)
            for q2, (q2eko, op) in zip((10.0, 11.0), ekoo.items()):
                assert q2 == q2eko
                np.testing.assert_allclose(v, op.operator)
            # getter
            with pytest.raises(ValueError):
                ekoo[12.0]
            with ekoo.operator(10.0) as op:
                np.testing.assert_allclose(v, op.operator)
            # overwrite
            vv = np.random.rand(2, 2)
            opvv = struct.Operator(operator=vv)
            ekoo[11.0] = opvv
            np.testing.assert_allclose(vv, ekoo[11.0].operator)

    def test_interpolator(self):
        nt, no = self._default_cards()
        txg = np.geomspace(0.1, 1.0, 5)
        ixg = np.geomspace(0.01, 1.0, 5)
        no["rotations"]["targetgrid"] = txg
        no["rotations"]["inputgrid"] = ixg
        with struct.EKO.create(nt, no) as ekoo:
            assert ekoo.interpolator(False, True).xgrid == interpolation.XGrid(txg)
            assert ekoo.interpolator(False, False).xgrid == interpolation.XGrid(ixg)

    def test_create(self, tmp_path):
        p = tmp_path / "eko.tar"
        with struct.EKO.create(*self._default_cards(), p) as _ekoo:
            pass
        assert p.exists()

    def test_copy(self, tmp_path):
        v = np.random.rand(2, 2)
        vv = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        p = tmp_path / "eko2.tar"
        with struct.EKO.create(*self._default_cards()) as eko1:
            eko1[10.0] = opv
            np.testing.assert_allclose(eko1[10.0].operator, v)
            eko1.deepcopy(p)
            assert p.exists()
            with struct.EKO.open(p) as eko2:
                np.testing.assert_allclose(eko1[10.0].operator, v)
                np.testing.assert_allclose(eko2[10.0].operator, v)
                opvv = struct.Operator(operator=vv)
                eko2[10.0] = opvv
                np.testing.assert_allclose(eko1[10.0].operator, v)
                np.testing.assert_allclose(eko2[10.0].operator, vv)
            # even after eko2 has be closed, eko1 has to remain the same
            np.testing.assert_allclose(eko1[10.0].operator, v)
        # even after eko1 has be closed, eko2 has to remain the same
        with struct.EKO.open(p) as eko2:
            np.testing.assert_allclose(eko2[10.0].operator, vv)


# class TestLegacy:
#     def test_items(self, fake_output):
#         """Test autodump, autoload, and manual unload."""
#         eko, fake_card = fake_output
#         for q2, op in fake_card["Q2grid"].items():
#             eko[q2] = output.Operator.from_dict(op)

#         q2 = next(iter(fake_card["Q2grid"]))

#         eko._operators[q2] = None
#         assert isinstance(eko[q2], struct.Operator)
#         assert isinstance(eko._operators[q2], struct.Operator)

#         del eko[q2]

#         assert eko._operators[q2] is None

#     def test_iter(self, fake_output):
#         """Test managed iteration."""
#         eko, fake_card = fake_output
#         for q2, op in fake_card["Q2grid"].items():
#             eko[q2] = output.Operator.from_dict(op)

#         q2prev = None
#         for q2, op in eko.items():
#             if q2prev is not None:
#                 assert eko._operators[q2prev] is None
#             assert isinstance(op, struct.Operator)
#             q2prev = q2

#     def test_context_operator(self, fake_output):
#         """Test automated handling through context."""
#         eko, fake_card = fake_output
#         for q2, op in fake_card["Q2grid"].items():
#             eko[q2] = output.Operator.from_dict(op)

#         q2 = next(iter(fake_card["Q2grid"]))

#         with eko.operator(q2) as op:
#             assert isinstance(op, struct.Operator)

#         assert eko._operators[q2] is None
