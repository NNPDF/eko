# -*- coding: utf-8 -*-
# Test NLO OME
import numpy as np

from eko.matching_conditions.nlo import (
    A_singlet_1,
    A_singlet_1_intrinsic,
    A_ns_1_intrinsic,
)

from eko.matching_conditions.operator_matrix_element import build_ome, A_non_singlet, A_singlet


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

def test_ome():
    # test that the matching is an identity when L=0 and not intrinsic
    N = 2
    L = 0.0
    a_s = 20
    sx = np.zeros(3, np.complex_)
    aNS = A_non_singlet(1, N, sx, L, False)
    aS = A_singlet(1, N, sx, L, False)

    for a in [aNS, aS]:
        for method in ["", "expanded", "exact"]:
            dim = len(a[0])
            assert len(a) == 1
            assert a[0].all() == np.zeros((dim,dim)).all()

            ome = build_ome(a, 1, a_s, method)
            assert ome.shape == (dim,dim)
            assert ome.all() == np.eye(dim).all()


    # test that the matching is not an identity when L=0 and intrinsic
    aNSi = A_non_singlet(1, N, sx, L, True)
    aSi = A_singlet(1, N, sx, L, True)
    for a in [aNSi, aSi]:
        for method in ["", "expanded"]:
            dim = len(a[0])
            # hh
            assert a[0,-1,-1] != 0.0
            # qh
            assert a[0,-2,-1] == 0.0
            ome = build_ome(a, 1, a_s, method)
            assert ome.shape == (dim,dim)
            assert ome[-1,-1] != 1.0
            assert ome[-2,-1] == 0.0
            assert ome[-1,-2] == 0.0
            assert ome[-2,-2] == 1.0

    # check gh for singlet
    assert aSi[0,0,-1] != 0.0
    assert ome[0,-1] != 0.0
