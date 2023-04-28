import numpy as np

from eko.constants import log2, zeta2, zeta3, zeta4, zeta5
from ekore import harmonics as h

from . import s, sm


def test_harmonic_definition():
    for power in [1, 2]:
        for k in range(1, 5 + 1):
            f = np.sum([1.0 / j**power for j in range(1, k + 1)])
            Sx = s(power, k)
            np.testing.assert_allclose(f, Sx)
            g = np.sum([(-1.0) ** j / j**power for j in range(1, k + 1)])
            np.testing.assert_allclose(g, sm(power, k, (-1) ** k == 1))


# TODO: CLEAN
# def test_harmonics_cache():
#     N = np.random.randint(1, high=100)
#     is_singlet = (-1) ** N == 1
#     S1 = h.S1(N)
#     S2 = h.S2(N)
#     S3 = h.S3(N)
#     S4 = h.S4(N)
#     S5 = h.S5(N)
#     Sm1 = h.Sm1(N, S1, is_singlet)
#     Sm2 = h.Sm2(N, S2, is_singlet)
#     sx = np.array([S1, S2, S3, S4, S5])
#     smx_test = np.array(
#         [
#             Sm1,
#             Sm2,
#             h.Sm3(N, S3, is_singlet),
#             h.Sm4(N, S4, is_singlet),
#             h.Sm5(N, S5, is_singlet),
#         ]
#     )
#     np.testing.assert_allclose(h.smx(N, sx, is_singlet), smx_test)
#     s3x_test = np.array(
#         [
#             h.S21(N, S1, S2),
#             h.S2m1(N, S2, Sm1, Sm2, is_singlet),
#             h.Sm21(N, S1, Sm1, is_singlet),
#             h.Sm2m1(N, S1, S2, Sm2),
#         ]
#     )
#     np.testing.assert_allclose(h.s3x(N, sx, smx_test, is_singlet), s3x_test)
#     Sm31 = h.Sm31(N, S1, Sm1, Sm2, is_singlet)
#     s4x_test = np.array(
#         [
#             h.S31(N, S1, S2, S3, S4),
#             h.S211(N, S1, S2, S3),
#             h.Sm22(N, S1, S2, Sm2, Sm31, is_singlet),
#             h.Sm211(N, S1, S2, Sm1, is_singlet),
#             Sm31,
#         ]
#     )
#     np.testing.assert_allclose(h.s4x(N, sx, smx_test, is_singlet), s4x_test)


# reference values coming fom mathematica
# and are computed doing an inverse mellin
# transformation
testN = [1, 2, 2 + 2j, 10 + 5j, 100]
refvals = {
    "Sm1": [-1.0, -0.5, -0.692917 - 0.000175788j, -0.693147 - 2.77406e-9j, -0.688172],
    "Sm2": [
        -1.0,
        -0.75,
        -0.822442 - 0.0000853585j,
        -0.822467 - 4.29516e-10j,
        -0.822418,
    ],
    "Sm3": [
        -1.0,
        -0.875,
        -0.901551 - 0.0000255879j,
        -0.901543 - 4.61382e-11j,
        -0.901542,
    ],
    "Sm4": [
        -1.0,
        -0.9375,
        -0.947039 - 4.84597e-6j,
        -0.947033 - 3.99567e-12j,
        -0.947033,
    ],
    "Sm5": [-1.0, -0.96875, -0.972122 - 1.13162e-7j, -0.97212 - 2.81097e-13j, -0.97212],
    "Sm21": [
        -1.0,
        -0.625,
        -0.751192 - 0.000147181j,
        -0.751286 - 1.17067e-9j,
        -0.751029,
    ],
}


def test_Sm21():
    for N, vals in zip(testN, refvals["Sm21"]):
        S1 = h.S1(N)
        S1mh = h.S1((N - 1) / 2)
        S1h = h.S1(N / 2)
        Sm1 = h.Sm1(N, S1, S1mh, S1h)
        np.testing.assert_allclose(h.Sm21(N, S1, Sm1), vals, atol=1e-06)


# TODO: CLEAN
# def test_Smx():
#     for j, N in enumerate(testN):
#         sx = h.sx(N)
#         smx = [
#             h.Sm1(N, sx[0]),
#             h.Sm2(N, sx[1]),
#             h.Sm3(N, sx[2]),
#             h.Sm4(N, sx[3]),
#             h.Sm5(N, sx[4]),
#         ]
#         for i, sm in enumerate(smx):
#             np.testing.assert_allclose(sm, refvals[f"Sm{i+1}"][j], atol=1e-06)


def test_smx_continuation():
    # test s_{-m} against a different analytic continuation
    N = np.random.rand() + 1j * np.random.rand()

    def dm(m):
        zeta_list = [
            zeta2,
            zeta3,
            zeta4,
            zeta5,
        ]
        if m == 1:
            return log2
        return (2 ** (m - 1) - 1) / (2 ** (m - 1)) * zeta_list[m - 2]

    def sm_complex(m, N):
        return ((-1) ** N) / 2**m * (s(m, N / 2) - s(m, (N - 1) / 2)) - dm(m)

    for j in range(1, 6):
        np.testing.assert_allclose(sm_complex(j, N), sm(j, N))
