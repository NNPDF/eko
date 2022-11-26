import numpy as np
import pytest

from eko import basis_rotation as br


def test_ad_projector():
    s = br.rotate_flavor_to_evolution[1]
    g = br.rotate_flavor_to_evolution[2]
    v3 = br.rotate_flavor_to_evolution[br.evol_basis.index("V3")]

    s_to_s = br.ad_projector((100, 100), nf=6)

    np.testing.assert_allclose(s @ s_to_s, s)
    np.testing.assert_allclose(g @ s_to_s, 0.0)
    np.testing.assert_allclose(v3 @ s_to_s, 0.0)

    g_to_s = br.ad_projector((21, 100), nf=6)

    np.testing.assert_allclose(s @ g_to_s, 0.0)
    np.testing.assert_allclose(g @ g_to_s, s)
    np.testing.assert_allclose(v3 @ g_to_s, 0.0)

    ns_m = br.ad_projector((br.non_singlet_pids_map["ns-"], 0), nf=6)

    np.testing.assert_allclose(s @ ns_m, 0.0, atol=1e-15)
    np.testing.assert_allclose(g @ ns_m, 0.0)
    np.testing.assert_allclose(v3 @ ns_m, v3)


def test_ad_projectors():
    for nf in range(3, 6 + 1):
        diag = np.array([0] * (1 + 6 - nf) + [1] * (1 + 2 * nf) + [0] * (6 - nf))
        identity = np.diag(diag)
        projs = br.ad_projectors(nf)

        # sum over diagonal projectors form an identity
        np.testing.assert_allclose(
            projs[0] + projs[3:].sum(axis=0),
            identity,
            atol=1e-15,
            err_msg=f"nf = {nf}",
        )


def test_intrinsic_unified_evol_labels():
    for nf in range(3, 6 + 1):
        labels = br.intrinsic_unified_evol_labels(nf)
        assert len(labels) == 14
    # errors
    with pytest.raises(IndexError):
        br.intrinsic_unified_evol_labels(7)
