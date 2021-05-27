import numpy as np

from eko import basis_rotation as br


def test_ad_projector():
    s = br.rotate_flavor_to_evolution[1]
    g = br.rotate_flavor_to_evolution[2]
    v3 = br.rotate_flavor_to_evolution[br.evol_basis.index("V3")]

    s_to_s = br.ad_projector("S_qq")

    np.testing.assert_allclose(s_to_s @ s, s)
    np.testing.assert_allclose(s_to_s @ g, 0.0)
    np.testing.assert_allclose(s_to_s @ v3, 0.0)

    g_to_s = br.ad_projector("S_gq")

    np.testing.assert_allclose(g_to_s @ s, g)
    np.testing.assert_allclose(g_to_s @ g, 0.0)
    np.testing.assert_allclose(g_to_s @ v3, 0.0)

    ns_m = br.ad_projector("NS_m")

    np.testing.assert_allclose(ns_m @ s, 0.0, atol=1e-15)
    np.testing.assert_allclose(ns_m @ g, 0.0)
    np.testing.assert_allclose(ns_m @ v3, v3)
