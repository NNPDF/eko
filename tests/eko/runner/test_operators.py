import numpy as np

from eko.io.items import Operator
from eko.io.struct import EKO
from eko.runner.operators import _retrieve, join


def test_retrieve(ekoparts: EKO):
    evhead, _evop = next(iter(ekoparts.parts.cache.items()))
    matchhead, _matchop = next(iter(ekoparts.parts_matching.cache.items()))

    els = _retrieve([evhead] * 5, ekoparts.parts, ekoparts.parts_matching)
    assert len(els) == 5
    assert all(isinstance(el, Operator) for el in els)

    els = _retrieve(
        [evhead, matchhead, matchhead], ekoparts.parts, ekoparts.parts_matching
    )
    assert len(els) == 3
    assert all(isinstance(el, Operator) for el in els)


def test_join(identity: Operator):
    """Just check a trivial property: product of identities is identity.

    In principle the join operation should respect all the properties of the
    matrix product, but there not so many sensible rank-4 operators.

    """
    linear_size = np.prod(identity.operator.shape[:2])
    for n in range(1, 8, 3):
        res = join([identity for _ in range(n)])
        assert res.error is None
        np.testing.assert_allclose(
            res.operator.reshape(linear_size, linear_size), np.eye(linear_size)
        )
