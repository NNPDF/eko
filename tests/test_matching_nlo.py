# -*- coding: utf-8 -*-
# Test NLO operator matrix elements
import numpy as np

from eko.matching_conditions.nlo import (
    A_singlet_1,
    A_singlet_1_intrinsic,
    A_ns_1_intrinsic,
)


def test_A_1_intrinsic():

    L = 0.0
    N = 2
    aS1 = A_singlet_1_intrinsic(N, L)
    # gluon momentum conservation
    np.testing.assert_allclose(aS1[2, 2] + aS1[0, 2], 0.0, atol=1e-8)


def test_A_1_shape():

    N = 2
    L = 3.0
    aNS1i = A_ns_1_intrinsic(N, L)
    aS1 = A_singlet_1(N, L)
    aS1i = A_singlet_1_intrinsic(N, L)

    assert aNS1i.shape == (2, 2)
    assert aS1.shape == (3, 3)
    assert aS1i.shape == (3, 3)

    # check q line equal to the h line for non intrinsic
    assert aS1[1].all() == aS1[2].all()

    # check intrisic hh is the same
    assert aNS1i[1, 1] == aS1i[2, 2]
