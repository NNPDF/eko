# -*- coding: utf-8 -*-
import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from eko import interpolation, output
from eko.output import struct


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

    def test_load_error(self, monkeypatch):
        # We might consider dropping this exception since np.load will always return a array (or fail on it's own)
        stream = io.BytesIO()
        monkeypatch.setattr(np, "load", lambda _: None)
        with pytest.raises(ValueError):
            struct.Operator.load(stream, False)


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


class TestLegacy:
    def test_items(self, fake_output):
        """Test autodump, autoload, and manual unload."""
        eko, fake_card = fake_output
        for q2, op in fake_card["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2 = next(iter(fake_card["Q2grid"]))

        eko._operators[q2] = None
        assert isinstance(eko[q2], struct.Operator)
        assert isinstance(eko._operators[q2], struct.Operator)

        del eko[q2]

        assert eko._operators[q2] is None

    def test_iter(self, fake_output):
        """Test managed iteration."""
        eko, fake_card = fake_output
        for q2, op in fake_card["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2prev = None
        for q2, op in eko.items():
            if q2prev is not None:
                assert eko._operators[q2prev] is None
            assert isinstance(op, struct.Operator)
            q2prev = q2

    def test_context_operator(self, fake_output):
        """Test automated handling through context."""
        eko, fake_card = fake_output
        for q2, op in fake_card["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2 = next(iter(fake_card["Q2grid"]))

        with eko.operator(q2) as op:
            assert isinstance(op, struct.Operator)

        assert eko._operators[q2] is None
