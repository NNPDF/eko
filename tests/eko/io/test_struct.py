import io
import pathlib
import tarfile

import numpy as np
import pytest
import yaml

from eko import EKO, interpolation
from eko.io import struct
from eko.io.items import Target
from tests.conftest import EKOFactory


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
        mu = 10.0
        nf = 5
        mugrid = [(mu, nf)]
        eko_factory.operator.mugrid = mugrid
        eko = eko_factory.get()
        assert hasattr(eko.theory_card.heavy, "masses")
        assert hasattr(eko.operator_card, "debug")
        np.testing.assert_allclose(eko.mu2grid, [mu**2])
        assert (mu**2, nf) in eko
        default_grid = eko.operator_card.xgrid
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
        nf = 5
        ep = (mu2, nf)
        mugrid = [(mu, nf)]
        eko_factory.operator.mugrid = mugrid
        eko = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        # approx
        eko[ep] = opv
        assert eko.approx((2 * mu2, nf)) is None
        assert eko.approx((mu2 + 1.0, nf), atol=2)[0] == mu2
        eko[(mu2 + 1.0, nf)] = opv
        with pytest.raises(ValueError):
            eko.approx((mu2 + 0.5, nf), atol=2)
        # iterate
        for mu2_, ep in zip((mu2, mu2 + 1.0), eko):
            assert mu2_ == ep[0]
            np.testing.assert_allclose(v, eko[(mu2, nf)].operator)
        for mu2_, (mu2eko, op) in zip((mu2, mu2 + 1.0), eko.items()):
            assert mu2_ == mu2eko[0]
            np.testing.assert_allclose(v, op.operator)
        # getter
        with pytest.raises(ValueError):
            eko[mu2 + 2.0, nf]
        with eko.operator(ep) as op:
            np.testing.assert_allclose(v, op.operator)
        # overwrite
        vv = np.random.rand(2, 2)
        opvv = struct.Operator(operator=vv)
        eko[mu2 + 1.0, nf] = opvv
        np.testing.assert_allclose(vv, eko[mu2 + 1.0, nf].operator)

    def test_copy(self, eko_factory: EKOFactory, tmp_path: pathlib.Path):
        mu = 10.0
        mu2 = mu**2
        nf = 5
        ep = (mu2, nf)
        mugrid = [(mu, nf)]
        eko_factory.operator.mugrid = mugrid
        eko1 = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        eko1[ep] = opv
        np.testing.assert_allclose(eko1[ep].operator, v)
        p = tmp_path / "eko2.tar"
        eko1.deepcopy(p)
        with EKO.edit(p) as eko2:
            np.testing.assert_allclose(eko1[ep].operator, v)
            np.testing.assert_allclose(eko2[ep].operator, v)
            vv = np.random.rand(2, 2)
            opvv = struct.Operator(operator=vv)
            eko2[ep] = opvv
            np.testing.assert_allclose(eko1[ep].operator, v)
            np.testing.assert_allclose(eko2[ep].operator, vv)
            # dump does not happen before closing, unless explicitly called, and
            # without a dump the path would be empty
            eko2.dump()
            eko2.unload()
            # try loading again
            eko2_ = struct.EKO.read(p)
            assert eko2.raw == eko2_.raw

    def test_items(self, eko_factory: EKOFactory):
        """Test autodump, autoload, and manual unload."""
        eko = eko_factory.get()
        v = np.random.rand(2, 2)
        opv = struct.Operator(operator=v)
        for ep in eko.operator_card.evolgrid:
            eko[ep] = opv

        ep = next(iter(eko))

        # unload
        eko.operators.cache[Target.from_ep(ep)] = None
        # test autoloading
        assert isinstance(eko[ep], struct.Operator)
        assert isinstance(eko.operators[Target.from_ep(ep)], struct.Operator)

        del eko[ep]

        assert eko.operators.cache[Target.from_ep(ep)] is None

    def test_iter(self, eko_factory: EKOFactory):
        """Test managed iteration."""
        eko_factory.operator.mugrid = [(3.0, 4), (20.0, 5), (300.0, 6)]
        eko = eko_factory.get()

        epprev = None
        for ep, op in eko.items():
            if epprev is not None:
                assert eko.operators[epprev] is None
            assert isinstance(op, struct.Operator)
            epprev = ep

    def test_context_operator(self, eko_factory: EKOFactory):
        """Test automated handling through context."""
        eko = eko_factory.get()
        ep = next(iter(eko))

        with eko.operator(ep) as op:
            assert isinstance(op, struct.Operator)

        assert eko.operators[ep] is None
