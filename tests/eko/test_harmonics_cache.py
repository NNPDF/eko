# -*- coding: utf-8 -*-
# Test eko.matching_conditions.OperatorMatrixElement
import numpy as np

from eko import harmonics as h


def test_HarmonicsCache():
    N = np.random.rand() + 1.0j * np.random.rand()
    Sm1 = h.Sm1(N)
    Sm2 = h.Sm2(N)
    S1 = h.S1(N)
    S2 = h.S2(N)
    S3 = h.S3(N)
    S4 = h.S4(N)
    sx = np.array([S1, S2, S3, S4, h.S5(N)])
    smx_test = np.array(
        [
            Sm1,
            Sm2,
            h.Sm3(N),
            h.Sm4(N),
            h.Sm5(N),
        ]
    )
    np.testing.assert_allclose(h.smx(N), smx_test)
    s3x_test = np.array(
        [
            h.S21(N, S1, S2),
            h.S2m1(N, S2, Sm1, Sm2),
            h.Sm21(N, S1, Sm1),
            h.Sm2m1(N, S1, S2, Sm2),
        ]
    )
    np.testing.assert_allclose(h.s3x(N, sx, smx_test), s3x_test)
    Sm31 = h.Sm31(N, S1, Sm1, Sm2)
    s4x_test = np.array(
        [
            h.S31(N, S1, S2, S4),
            h.S211(N, S1, S2, S3),
            h.Sm22(N, S1, Sm2, Sm31),
            h.Sm211(N, S1, S2, Sm1),
            Sm31,
        ]
    )
    np.testing.assert_allclose(h.s4x(N, sx, smx_test), s4x_test)
