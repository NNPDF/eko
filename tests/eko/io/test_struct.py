import io
import pathlib
import tarfile
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import yaml

from eko import EKO, interpolation
from eko.io import dictlike, struct
from tests.conftest import EKOFactory


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
        with pytest.raises(ValueError, match="Blub.bla"):
            struct.EKO.create(no_tar_path)
        # try to overwrite an existing file
        exists_path = tmp_path / "Blub.tar"
        with tarfile.open(exists_path, "w") as tar:
            tar.add(no_tar_path)
        with pytest.raises(FileExistsError, match="Blub.tar"):
            struct.EKO.create(exists_path)

    def test_load_error(self, tmp_path):
        # try to read from a non-tar path
        no_tar_path = tmp_path / "Blub.tar"
        no_tar_path.write_text("Blub", encoding="utf-8")
        with pytest.raises(ValueError):
            struct.EKO.read(no_tar_path)

    def test_properties(self, eko_factory: EKOFactory):
        mugrid = np.array([10.0])
        eko_factory.operator.mugrid = mugrid
        eko = eko_factory.get()
        assert hasattr(eko.theory_card, "quark_masses")
        assert hasattr(eko.operator_card, "debug")
        np.testing.assert_allclose(eko.mu2grid, mugrid**2)
        assert mugrid[0] ** 2 in eko
        default_grid = eko.operator_card.rotations.xgrid
        assert eko.xgrid == default_grid
        xg = interpolation.XGrid([0.1, 1.0])
        eko.xgrid = xg
        assert eko.xgrid == xg
        assert "metadata" in eko.raw
        # check we can dump and reload
        stream = io.StringIO()
        yaml.safe_dump(eko.raw, stream)
        stream.seek(0)
        raw_eko = yaml.safe_load(stream)
        assert "metadata" in raw_eko

    def test_ops(self, eko_factory: EKOFactory):
        mu = 10.0
        mu2 = mu**2
        mugrid = np.array([mu])
        eko_factory.operator.mugrid = mugrid
        eko = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        # try setting not an operator
        with pytest.raises(ValueError):
            eko[mu2] = "bla"
        # approx
        eko[mu2] = opv
        assert eko.approx(2 * mu2) is None
        assert eko.approx(mu2 + 1.0, atol=2) == mu2
        eko[mu2 + 1.0] = opv
        with pytest.raises(ValueError):
            eko.approx(mu2 + 0.5, atol=2)
        # iterate
        for q2, q2eko in zip((mu2, mu2 + 1.0), eko):
            assert q2 == q2eko
            np.testing.assert_allclose(v, eko[q2].operator)
        for q2, (q2eko, op) in zip((mu2, mu2 + 1.0), eko.items()):
            assert q2 == q2eko
            np.testing.assert_allclose(v, op.operator)
        # getter
        with pytest.raises(ValueError):
            eko[mu2 + 2.0]
        with eko.operator(mu2) as op:
            np.testing.assert_allclose(v, op.operator)
        # overwrite
        vv = np.random.rand(2, 2)
        opvv = struct.Operator(operator=vv)
        eko[mu2 + 1.0] = opvv
        np.testing.assert_allclose(vv, eko[mu2 + 1.0].operator)

    def test_copy(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        mu = 10.0
        mu2 = mu**2
        mugrid = np.array([mu])
        eko_factory.operator.mugrid = mugrid
        eko1 = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        eko1[mu2] = opv
        np.testing.assert_allclose(eko1[mu2].operator, v)
        p = tmp_path / "eko2.tar"
        eko1.deepcopy(p)
        with EKO.edit(p) as eko2:
            np.testing.assert_allclose(eko1[mu2].operator, v)
            np.testing.assert_allclose(eko2[mu2].operator, v)
            vv = np.random.rand(2, 2)
            opvv = struct.Operator(operator=vv)
            eko2[mu2] = opvv
            np.testing.assert_allclose(eko1[mu2].operator, v)
            np.testing.assert_allclose(eko2[mu2].operator, vv)
            # dump does not happen before closing, unless explicitly called, and
            # without a dump the path would be empty
            eko2.dump()
            eko2.unload()
            # try loading again
            eko2_ = struct.EKO.read(p)
            assert eko2.raw == eko2_.raw


class TestLegacy:
    def test_items(self, eko_factory: EKOFactory):
        """Test autodump, autoload, and manual unload."""
        eko = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        for mu2 in eko.operator_card.mu2grid:
            eko[mu2] = opv

        mu2 = next(iter(eko.mu2grid))

        # unload
        eko._operators[mu2] = None
        # test autoloading
        assert isinstance(eko[mu2], struct.Operator)
        assert isinstance(eko._operators[mu2], struct.Operator)

        del eko[mu2]

        assert eko._operators[mu2] is None

    def test_iter(self, eko_factory):
        """Test managed iteration."""
        eko_factory.operator.mugrid = np.array([5.0, 20.0, 100.0])
        eko = eko_factory.get()

        mu2prev = None
        for mu2, op in eko.items():
            if mu2prev is not None:
                assert eko._operators[mu2prev] is None
            assert isinstance(op, struct.Operator)
            mu2prev = mu2

    def test_context_operator(self, eko_factory):
        """Test automated handling through context."""
        eko = eko_factory.get()
        mu2 = eko.mu2grid[0]

        with eko.operator(mu2) as op:
            assert isinstance(op, struct.Operator)

        assert eko._operators[mu2] is None
