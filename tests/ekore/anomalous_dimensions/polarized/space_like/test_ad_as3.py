# Test NNLO Polarized splitting functions
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like.as3 as as3
import ekore.anomalous_dimensions.unpolarized.space_like.as3 as as3_unpol
from ekore import harmonics

nf = 5


def test_gluon_momentum():
    # gluon momentum
    N = complex(2.0, 0.0)
    cache = harmonics.cache.reset()
    np.testing.assert_allclose(
        as3.gamma_qg(N, nf, cache) + as3.gamma_gg(N, nf, cache), 9.26335, rtol=7e-4
    )


def test_qg_helicity_conservation():
    N = complex(1.0, 0.0)
    cache = harmonics.cache.reset()
    np.testing.assert_almost_equal(as3.gamma_qg(N, nf, cache), 0.00294317)


def test_ns_sea():
    ref_moments = [
        -220 / 3,
        -103175 / 34992,
        -4653353 / 5467500,
        -7063530941 / 17425497600,
        -218695344199 / 911421315000,
    ]
    for i, mom in enumerate(ref_moments):
        N = 1 + 2 * i
        cache = harmonics.cache.reset()
        np.testing.assert_allclose(-as3.gamma_nss(N, nf, cache), mom * nf, rtol=7e-7)


def test_ns():
    N = complex(3.45, 0.0)
    cache = harmonics.cache.reset()
    np.testing.assert_allclose(
        as3.gamma_nsv(N, nf, cache),
        as3.gamma_nsm(N, nf, cache) + as3.gamma_nss(N, nf, cache),
    )
    np.testing.assert_allclose(
        as3_unpol.gamma_nsm(N, nf, cache), as3.gamma_nsp(N, nf, cache)
    )
    np.testing.assert_allclose(
        as3_unpol.gamma_nsp(N, nf, cache), as3.gamma_nsm(N, nf, cache)
    )
