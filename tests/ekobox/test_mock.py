import numpy as np

from ekobox import mock


def test_eko_identity():
    for s in ((1, 2, 2, 2, 2), (1, 3, 3, 3, 3)):
        i = mock.eko_identity(s)
        assert s == i.shape
        # is identity?
        f = np.random.rand(*s[-2:])
        g = np.einsum("qkbja,ja->qkb", i, f)
        np.testing.assert_allclose(g[0], f)
