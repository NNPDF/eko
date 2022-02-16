# -*- coding: utf-8 -*-

import numpy as np

from eko.scale_variations import b


def test_ns_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_ns = np.random.rand(order + 1)
    L = 0
    nf = 5
    a_s = 0.35
    ker_test = np.random.random(1)
    np.testing.assert_allclose(
        b.non_singlet_variation(ker_test, gamma_ns, a_s, order, nf, L), ker_test
    )


def test_singlet_sv_dispacher():
    """Test to identity"""
    order = 2
    gamma_singlet = np.random.rand(order + 1, 2, 2)
    L = 0
    nf = 5
    a_s = 0.35
    ker_test = np.random.rand(2, 2)
    np.testing.assert_allclose(
        b.singlet_variation(ker_test, gamma_singlet, a_s, order, nf, L), ker_test
    )
