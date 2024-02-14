import numpy as np

from eko import basis_rotation as br
from eko.beta import beta_qcd_as2
from eko.couplings import CouplingEvolutionMethod, Couplings, CouplingsInfo
from eko.kernels import non_singlet, singlet
from eko.quantities.heavy_quarks import QuarkMassScheme
from eko.scale_variations import Modes, expanded, exponentiated
from ekore.anomalous_dimensions.unpolarized.space_like import gamma_ns, gamma_singlet


def test_modes():
    assert Modes.expanded.name == "expanded"
    assert Modes.exponentiated.name == "exponentiated"
    assert Modes.unvaried.name == "unvaried"
    assert Modes.expanded.value == 3
    assert Modes.exponentiated.value == 2
    assert Modes.unvaried.value == 1


def test_ns_sv_dispacher():
    """Test to identity"""
    order = (4, 0)
    gamma_ns = np.random.rand(order[0])
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        expanded.non_singlet_variation(gamma_ns, a_s, order, nf, L), 1.0
    )


def test_ns_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_ns = np.random.rand(order[0], order[1])
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.non_singlet_variation_qed(
                gamma_ns, a_s, a_em, alphaem_running, order, nf, L
            ),
            1.0,
        )


def test_singlet_sv_dispacher():
    """Test to identity"""
    order = (4, 0)
    gamma_singlet = np.random.rand(order[0], 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    np.testing.assert_allclose(
        expanded.singlet_variation(gamma_singlet, a_s, order, nf, L, 2), np.eye(2)
    )


def test_singlet_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_singlet = np.random.rand(order[0], order[1], 4, 4)
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.singlet_variation_qed(
                gamma_singlet, a_s, a_em, alphaem_running, order, nf, L
            ),
            np.eye(4),
        )


def test_valence_sv_dispacher_qed():
    """Test to identity"""
    order = (4, 2)
    gamma_valence = np.random.rand(order[0], order[1], 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    a_em = 0.01
    for alphaem_running in [True, False]:
        np.testing.assert_allclose(
            expanded.valence_variation_qed(
                gamma_valence, a_s, a_em, alphaem_running, order, nf, L
            ),
            np.eye(2),
        )
