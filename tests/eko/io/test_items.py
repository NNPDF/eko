import io

import lz4.frame
import numpy as np
import pytest

from eko.io.items import Evolution, Matching, Operator, Target
from eko.matchings import Atlas
from eko.quantities.heavy_quarks import MatchingScales


def test_evolution():
    tc = Atlas(MatchingScales([1.5, 5.0, 170.0]), (1.65, 4))
    p = tc.matched_path((30, 5))
    he = Evolution.from_atlas(p[0], True)
    assert he.origin == 1.65
    assert he.target == 5
    assert he.nf == 4
    assert he.cliff
    assert he.as_atlas == p[0]


def test_matching():
    tc = Atlas(MatchingScales([1.5, 5.0, 170.0]), (1.65, 4))
    p = tc.matched_path((30, 5))
    hm = Matching.from_atlas(p[1])
    assert hm.hq == 5
    assert hm.scale == 5.0
    assert hm.as_atlas == p[1]


def test_target():
    ep = (10.0, 4)
    ht = Target.from_ep(ep)
    assert ht.scale == 10.0
    assert ht.nf == 4
    assert ht.ep == ep


class TestOperator:
    def test_value_only(self):
        v = np.random.rand(2, 2)
        opv = Operator(operator=v)
        assert opv.error is None
        stream = io.BytesIO()
        opv.save(stream)
        stream.seek(0)
        opv_ = Operator.load(stream)
        np.testing.assert_allclose(opv.operator, opv_.operator)
        np.testing.assert_allclose(v, opv_.operator)
        assert opv_.error is None

    def test_value_and_error(self):
        v, e = np.random.rand(2, 2, 2)
        opve = Operator(operator=v, error=e)
        stream = io.BytesIO()
        opve.save(stream)
        stream.seek(0)
        opve_ = Operator.load(stream)
        np.testing.assert_allclose(opve.operator, opve_.operator)
        np.testing.assert_allclose(v, opve_.operator)
        np.testing.assert_allclose(opve.error, opve_.error)
        np.testing.assert_allclose(e, opve_.error)

    def test_load_error_is_not_lz4(self, monkeypatch):
        stream = io.BytesIO()
        with pytest.raises(RuntimeError, match="LZ4"):
            Operator.load(stream)

    def test_load_error(self, monkeypatch):
        # TODO see the other TODO at Operator.load
        v, e = np.random.rand(2, 2, 2)
        opve = Operator(operator=v, error=e)
        stream = io.BytesIO()
        opve.save(stream)
        stream.seek(0)
        monkeypatch.setattr(np, "load", lambda _: None)
        with pytest.raises(ValueError):
            Operator.load(stream)
