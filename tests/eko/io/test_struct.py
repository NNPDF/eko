import io
import pathlib
import tarfile
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from eko import interpolation
from eko.io import dictlike, runcards, struct
from ekobox import cards


@dataclass
class MyDictLike(dictlike.DictLike):
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
            x=[0.1, 1.0],
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
        # We might consider dropping this exception since np.load will always
        # return a array (or fail on it's own)
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


class TestEKO:
    def test_new_error(self, tmp_path: pathlib.Path):
        # try to write to a file different from bla
        no_tar_path = tmp_path / "Blub.bla"
        no_tar_path.touch()
        with pytest.raises(ValueError):
            struct.EKO.create(no_tar_path)
        # try to overwrite an existing file
        exists_path = tmp_path / "Blub.tar"
        with tarfile.open(exists_path, "w") as tar:
            tar.add(no_tar_path)
        with pytest.raises(FileExistsError):
            struct.EKO.create(exists_path)

    def test_load_error(self, tmp_path):
        # try to read from a non-tar path
        no_tar_path = tmp_path / "Blub.tar"
        no_tar_path.write_text("Blub", encoding="utf-8")
        with pytest.raises(ValueError):
            struct.EKO.read(no_tar_path)

    def test_properties(self):
        eko = struct.EKO.new(*self._default_cards())
        assert "mc" in eko.theory_card
        assert "debug" in eko.operator_card
        np.testing.assert_allclose(eko.mu2grid, np.array([10.0]))
        assert 10.0 in eko
        default_grid = interpolation.XGrid(eko.operator_card["rotations"]["xgrid"])
        assert eko.xgrid == default_grid
        for use_target in (True, False):
            assert eko.interpolator(False, use_target).xgrid == default_grid
        xg = interpolation.XGrid([0.1, 1.0])
        eko.xgrid = xg
        assert eko.xgrid == xg
        assert "debug" in eko.raw
        # check we can dump and reload
        stream = io.StringIO()
        yaml.safe_dump(eko.raw, stream)
        stream.seek(0)
        raw_eko = yaml.safe_load(stream)
        assert "debug" in raw_eko

    def test_ops(self):
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        eko = struct.EKO.new(*self._default_cards())
        # try setting not an operator
        with pytest.raises(ValueError):
            eko[10.0] = "bla"
        # approx
        eko[10.0] = opv
        assert eko.approx(20.0) is None
        assert eko.approx(11.0, atol=2) == 10.0
        eko[11.0] = opv
        with pytest.raises(ValueError):
            eko.approx(10.5, atol=2)
        # iterate
        for q2, q2eko in zip((10.0, 11.0), eko):
            assert q2 == q2eko
            np.testing.assert_allclose(v, eko[q2].operator)
        for q2, (q2eko, op) in zip((10.0, 11.0), eko.items()):
            assert q2 == q2eko
            np.testing.assert_allclose(v, op.operator)
        # getter
        with pytest.raises(ValueError):
            eko[12.0]
        with eko.operator(10.0) as op:
            np.testing.assert_allclose(v, op.operator)
        # overwrite
        vv = np.random.rand(2, 2)
        opvv = struct.Operator(operator=vv)
        eko[11.0] = opvv
        np.testing.assert_allclose(vv, eko[11.0].operator)

    def test_interpolator(self):
        nt, no = self._default_cards()
        txg = np.geomspace(0.1, 1.0, 5)
        ixg = np.geomspace(0.01, 1.0, 5)
        no["rotations"]["targetgrid"] = txg
        no["rotations"]["inputgrid"] = ixg
        eko = struct.EKO.new(nt, no)
        # Targetgrid and inputgrid should not be anymore in the opcard
        # They are not used anymore
        assert eko.interpolator(False, True).xgrid == interpolation.XGrid(
            no["rotations"]["xgrid"]
        )
        assert eko.interpolator(False, False).xgrid == interpolation.XGrid(
            no["rotations"]["xgrid"]
        )
        # However you can esplicitly set them
        eko.rotations._targetgrid = interpolation.XGrid(txg)
        eko.rotations._inputgrid = interpolation.XGrid(ixg)
        assert eko.interpolator(False, True).xgrid == interpolation.XGrid(txg)
        assert eko.interpolator(False, False).xgrid == interpolation.XGrid(ixg)

    def test_copy(self, tmp_path):
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        eko1 = struct.EKO.new(*self._default_cards())
        eko1[10.0] = opv
        np.testing.assert_allclose(eko1[10.0].operator, v)
        p = tmp_path / "eko2.tar"
        eko2 = eko1.deepcopy(p)
        np.testing.assert_allclose(eko1[10.0].operator, v)
        np.testing.assert_allclose(eko2[10.0].operator, v)
        vv = np.random.rand(2, 2)
        opvv = struct.Operator(operator=vv)
        eko2[10.0] = opvv
        np.testing.assert_allclose(eko1[10.0].operator, v)
        np.testing.assert_allclose(eko2[10.0].operator, vv)
        # try loading again
        eko2_ = struct.EKO.load(p)
        assert eko2.raw == eko2_.raw

    def test_extract(self, tmp_path):
        p = tmp_path / "test.tar"
        eko = struct.EKO.new(*self._default_cards(), p)
        # check theory file
        t = struct.EKO.extract(p, struct.THEORYFILE)
        assert isinstance(t, str)
        tt = yaml.safe_load(io.StringIO(t))
        assert tt == eko.theory_card
        # try a wrong file
        with pytest.raises(KeyError):
            t = struct.EKO.extract(p, "Blub.bla")


class TestLegacy:
    def test_items(self, fake_output):
        """Test autodump, autoload, and manual unload."""
        eko, fake_card = fake_output
        for q2, op in fake_card["Q2grid"].items():
            eko[q2] = eko.io.Operator.from_dict(op)

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
            eko[q2] = eko.io.Operator.from_dict(op)

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
            eko[q2] = eko.io.Operator.from_dict(op)

        q2 = next(iter(fake_card["Q2grid"]))

        with eko.operator(q2) as op:
            assert isinstance(op, struct.Operator)

        assert eko._operators[q2] is None
