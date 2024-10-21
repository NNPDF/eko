# Test NNLO Polarized splitting functions
import numpy as np

import ekore.anomalous_dimensions.polarized.space_like.as2 as as2
import ekore.anomalous_dimensions.polarized.space_like.as3 as as3
import ekore.anomalous_dimensions.unpolarized.space_like.as3 as as3_unpol
from eko import beta
from eko.constants import zeta2, zeta3
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
        (20 / 9)
        * (
            23 - 12 * zeta2 - 16 * zeta3
        ),  # reference value from eq 27 of :cite:`Moch:2015usa`
        -0.0004286694101508916 * (-12139.872394862155 - 21888 * -0.6738675265146354),
        0.5486968449931413 * (3.844116190831125 + 6.624 * -0.712742410330442),
        1.1187158441230356 + 1.927437641723356 * -0.7278824381560618,
        0.6893728427287487 + 1.1939643347050755 * -0.7354233751216146,
    ]
    for i, mom in enumerate(ref_moments):
        N = 1 + 2 * i
        cache = harmonics.cache.reset()
        np.testing.assert_allclose(-as3.gamma_nss(N, nf, cache), mom * nf, rtol=5e-7)


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


def test_axial_anomaly():
    # violation of the axial current conservation happens only through loops
    N = complex(1.0, 0.0)
    cache = harmonics.cache.reset()
    np.testing.assert_allclose(
        as3.gamma_gg(N, nf, cache), -beta.beta_qcd_as4(nf), rtol=2e-5
    )
    np.testing.assert_allclose(
        as3.gamma_ps(N, nf, cache), -2 * nf * as2.gamma_gq(N, nf, cache), rtol=3e-6
    )
