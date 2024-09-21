"""The unpolarized, space-like |N3LO| gluon-quark |OME|."""

import numba as nb
import numpy as np

from .....harmonics import cache as c


@nb.njit(cache=True)
def A_gq(n, cache, nf, L):  # pylint: disable=too-many-locals
    r"""Compute the |N3LO| singlet |OME| :math:`A_{gq}^{S,(3)}(N)`.

    The expression is presented in :cite:`Ablinger_2014` :eqref:`6.3`.

    When using the code, please cite the complete list of references
    available in :mod:`~ekore.operator_matrix_elements.unpolarized.space_like.as3`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache: numpy.ndarray
        Harmonic sum cache
    nf : int
        number of active flavor below the threshold
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        :math:`A_{gq}^{S,(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    Sm1 = c.get(c.Sm1, cache, n, is_singlet=True)
    S2 = c.get(c.S2, cache, n)
    Sm2 = c.get(c.Sm2, cache, n, is_singlet=True)
    S3 = c.get(c.S3, cache, n)
    S21 = c.get(c.S21, cache, n)
    S2m1 = c.get(c.S2m1, cache, n, is_singlet=True)
    Sm21 = c.get(c.Sm21, cache, n, is_singlet=True)
    Sm2m1 = c.get(c.Sm2m1, cache, n, is_singlet=True)
    Sm3 = c.get(c.Sm3, cache, n, is_singlet=True)
    S4 = c.get(c.S4, cache, n)
    S31 = c.get(c.S31, cache, n)
    S211 = c.get(c.S211, cache, n)
    Sm22 = c.get(c.Sm22, cache, n, is_singlet=True)
    Sm211 = c.get(c.Sm211, cache, n, is_singlet=True)
    Sm31 = c.get(c.Sm31, cache, n, is_singlet=True)
    Sm4 = c.get(c.Sm4, cache, n, is_singlet=True)
    a_gq_l0 = (
        0.3333333333333333
        * (
            (
                -0.06584362139917696
                * (
                    718.0
                    + 2495.0 * n
                    + 3608.0 * np.power(n, 2)
                    + 2944.0 * np.power(n, 3)
                    + 1364.0 * np.power(n, 4)
                    + 359.0 * np.power(n, 5)
                )
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 4))
            + (
                1.1851851851851851
                * (
                    8.0
                    + 25.0 * n
                    + 23.0 * np.power(n, 2)
                    + 4.0 * np.power(n, 3)
                    + 4.0 * np.power(n, 4)
                )
                * S1
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3))
            - (
                0.5925925925925926
                * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            - (
                0.5925925925925926
                * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                * S2
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            + nf
            * (
                (
                    0.26337448559670784
                    * (
                        394.0
                        + 1388.0 * n
                        + 1961.0 * np.power(n, 2)
                        + 1540.0 * np.power(n, 3)
                        + 824.0 * np.power(n, 4)
                        + 197.0 * np.power(n, 5)
                    )
                )
                / ((-1.0 + n) * n * np.power(1.0 + n, 4))
                + (
                    2.3703703703703702
                    * (
                        8.0
                        + 25.0 * n
                        + 23.0 * np.power(n, 2)
                        + 4.0 * np.power(n, 3)
                        + 4.0 * np.power(n, 4)
                    )
                    * S1
                )
                / ((-1.0 + n) * n * np.power(1.0 + n, 3))
                - (
                    1.1851851851851851
                    * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                    * np.power(S1, 2)
                )
                / ((-1.0 + n) * n * np.power(1.0 + n, 2))
                - (
                    1.1851851851851851
                    * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                    * S2
                )
                / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    59.835721401722026
                    + 0.5925925925925926 * np.power(S1, 3)
                    + 1.7777777777777777 * S1 * S2
                    + 1.1851851851851851 * S3
                    + nf
                    * (
                        -34.19184080098401
                        + 1.1851851851851851 * np.power(S1, 3)
                        + 3.5555555555555554 * S1 * S2
                        + 2.3703703703703702 * S3
                    )
                )
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                1.0684950250307503
                * (
                    192.0
                    + 608.0 * n
                    + 436.0 * np.power(n, 2)
                    - 280.0 * np.power(n, 3)
                    - 33.0 * np.power(n, 4)
                    + 436.0 * np.power(n, 5)
                    + 730.0 * np.power(n, 6)
                    + 484.0 * np.power(n, 7)
                    + 115.0 * np.power(n, 8)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * (2.0 + n)
            )
            + (
                0.00205761316872428
                * (
                    718848.0
                    + 3.036672e6 * n
                    + 7.448064e6 * np.power(n, 2)
                    + 3.6681856e7 * np.power(n, 3)
                    + 9.6114752e7 * np.power(n, 4)
                    + 9.0199968e7 * np.power(n, 5)
                    - 3.1178992e7 * np.power(n, 6)
                    - 1.39149336e8 * np.power(n, 7)
                    - 1.06346044e8 * np.power(n, 8)
                    + 2.390668e7 * np.power(n, 9)
                    + 1.19019157e8 * np.power(n, 10)
                    + 1.13250363e8 * np.power(n, 11)
                    + 4.9867573e7 * np.power(n, 12)
                    + 794307.0 * np.power(n, 13)
                    - 1.1396201e7 * np.power(n, 14)
                    - 6.177407e6 * np.power(n, 15)
                    - 1.469301e6 * np.power(n, 16)
                    - 138495.0 * np.power(n, 17)
                )
            )
            / (
                (-2.0 + n)
                * np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 4)
            )
            - (
                0.09876543209876543
                * (
                    -72.0
                    - 456.0 * n
                    - 847.0 * np.power(n, 2)
                    - 694.0 * np.power(n, 3)
                    - 343.0 * np.power(n, 4)
                    - 216.0 * np.power(n, 5)
                    + 4.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3))
            + (
                0.49382716049382713
                * (-6.0 + 11.0 * n + np.power(n, 2) + 4.0 * np.power(n, 3))
                * np.power(S1, 3)
            )
            / ((-1.0 + n) * np.power(n, 2) * (1.0 + n))
            - (
                0.09876543209876543
                * (
                    -5184.0
                    - 7776.0 * n
                    - 8496.0 * np.power(n, 2)
                    - 6172.0 * np.power(n, 3)
                    + 21932.0 * np.power(n, 4)
                    + 14047.0 * np.power(n, 5)
                    - 14788.0 * np.power(n, 6)
                    - 20968.0 * np.power(n, 7)
                    - 8170.0 * np.power(n, 8)
                    + 297.0 * np.power(n, 9)
                    + 3042.0 * np.power(n, 10)
                    + 1132.0 * np.power(n, 11)
                )
                * S2
            )
            / (
                (-2.0 + n)
                * np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * (2.0 + n)
            )
            + S1
            * (
                (
                    0.06584362139917696
                    * (
                        -108.0
                        - 1008.0 * n
                        - 4161.0 * np.power(n, 2)
                        - 6041.0 * np.power(n, 3)
                        - 1186.0 * np.power(n, 4)
                        + 5051.0 * np.power(n, 5)
                        + 5986.0 * np.power(n, 6)
                        + 3236.0 * np.power(n, 7)
                        + 1031.0 * np.power(n, 8)
                    )
                )
                / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4))
                + (
                    0.2962962962962963
                    * (
                        18.0
                        + 341.0 * n
                        + 546.0 * np.power(n, 2)
                        + 293.0 * np.power(n, 3)
                        + 166.0 * np.power(n, 4)
                    )
                    * S2
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2))
            )
            - (
                0.5925925925925926
                * (70.0 + 111.0 * n + 64.0 * np.power(n, 2) + 35.0 * np.power(n, 3))
                * S21
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            + (
                0.19753086419753085
                * (
                    -876.0
                    - 1532.0 * n
                    - 1631.0 * np.power(n, 2)
                    + 434.0 * np.power(n, 3)
                    + 828.0 * np.power(n, 4)
                    + 774.0 * np.power(n, 5)
                    + 275.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            - (
                10.666666666666666
                * (
                    64.0
                    + 120.0 * n
                    - 16.0 * np.power(n, 2)
                    - 2.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 5)
                    + 3.0 * np.power(n, 6)
                    + 3.0 * np.power(n, 7)
                    + np.power(n, 8)
                )
                * Sm2
            )
            / (
                (-2.0 + n)
                * np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    -141.50943229384302
                    + 0.37037037037037035 * np.power(S1, 4)
                    - 8.444444444444445 * np.power(S1, 2) * S2
                    - 0.6666666666666666 * np.power(S2, 2)
                    - 3.5555555555555554 * S211
                    - (128.0 * S2m1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + S1
                    * (
                        42.73980100123002
                        + 3.5555555555555554 * S21
                        - 11.25925925925926 * S3
                    )
                    + 10.666666666666666 * S31
                    - 19.11111111111111 * S4
                    + (128.0 * S2 * Sm1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    - (128.0 * Sm1 * Sm2) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + (128.0 * Sm2m1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + (42.666666666666664 * Sm3)
                    / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                )
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
        + 2.0
        * (
            (
                -0.5342475125153752
                * (
                    1120.0
                    - 172.0 * n
                    + 1084.0 * np.power(n, 2)
                    - 253.0 * np.power(n, 3)
                    + 65.0 * np.power(n, 4)
                    + 249.0 * np.power(n, 5)
                    + 19.0 * np.power(n, 6)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                0.03292181069958848
                * (
                    -6912.0
                    - 47232.0 * n
                    - 441984.0 * np.power(n, 2)
                    - 1.844224e6 * np.power(n, 3)
                    - 3.89016e6 * np.power(n, 4)
                    - 4.766232e6 * np.power(n, 5)
                    - 3.300816e6 * np.power(n, 6)
                    - 574564.0 * np.power(n, 7)
                    + 1.086304e6 * np.power(n, 8)
                    + 625443.0 * np.power(n, 9)
                    - 386836.0 * np.power(n, 10)
                    - 504489.0 * np.power(n, 11)
                    - 121050.0 * np.power(n, 12)
                    + 88963.0 * np.power(n, 13)
                    + 76658.0 * np.power(n, 14)
                    + 23287.0 * np.power(n, 15)
                    + 2596.0 * np.power(n, 16)
                )
            )
            / (
                (-2.0 + n)
                * np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                0.04938271604938271
                * (
                    1152.0
                    - 4752.0 * n
                    - 17128.0 * np.power(n, 2)
                    - 7888.0 * np.power(n, 3)
                    + 16190.0 * np.power(n, 4)
                    + 22753.0 * np.power(n, 5)
                    + 12689.0 * np.power(n, 6)
                    + 5119.0 * np.power(n, 7)
                    + 1821.0 * np.power(n, 8)
                    + 284.0 * np.power(n, 9)
                )
                * np.power(S1, 2)
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.19753086419753085
                * (
                    -150.0
                    - 1.0 * n
                    + 165.0 * np.power(n, 2)
                    + 49.0 * np.power(n, 3)
                    + 23.0 * np.power(n, 4)
                    + 4.0 * np.power(n, 5)
                )
                * np.power(S1, 3)
            )
            / (np.power(-1.0 + n, 2) * np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                0.14814814814814814
                * (
                    4608.0
                    + 5984.0 * n
                    + 2528.0 * np.power(n, 2)
                    + 10352.0 * np.power(n, 3)
                    + 10204.0 * np.power(n, 4)
                    + 434.0 * np.power(n, 5)
                    - 1853.0 * np.power(n, 6)
                    - 759.0 * np.power(n, 7)
                    - 265.0 * np.power(n, 8)
                    + 81.0 * np.power(n, 9)
                    + 78.0 * np.power(n, 10)
                )
                * S2
            )
            / (
                (-2.0 + n)
                * np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + S1
            * (
                (
                    -0.03292181069958848
                    * (
                        3456.0
                        + 33984.0 * n
                        + 39648.0 * np.power(n, 2)
                        - 118240.0 * np.power(n, 3)
                        - 256840.0 * np.power(n, 4)
                        - 15780.0 * np.power(n, 5)
                        + 382214.0 * np.power(n, 6)
                        + 482244.0 * np.power(n, 7)
                        + 299724.0 * np.power(n, 8)
                        + 120121.0 * np.power(n, 9)
                        + 39379.0 * np.power(n, 10)
                        + 10131.0 * np.power(n, 11)
                        + 1207.0 * np.power(n, 12)
                    )
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 4)
                    * np.power(1.0 + n, 4)
                    * np.power(2.0 + n, 3)
                )
                + (
                    0.5925925925925926
                    * (
                        -66.0
                        - 7.0 * n
                        - 283.0 * np.power(n, 2)
                        + 19.0 * np.power(n, 3)
                        + 72.0 * np.power(n, 4)
                        + 54.0 * np.power(n, 5)
                        + 31.0 * np.power(n, 6)
                    )
                    * S2
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * (2.0 + n)
                )
            )
            + (
                1.7777777777777777
                * (
                    88.0
                    - 10.0 * n
                    + 81.0 * np.power(n, 2)
                    + 4.0 * np.power(n, 3)
                    + 15.0 * np.power(n, 4)
                    + 14.0 * np.power(n, 5)
                )
                * S21
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                0.09876543209876543
                * (
                    24.0
                    - 356.0 * n
                    - 2066.0 * np.power(n, 2)
                    + 725.0 * np.power(n, 3)
                    + 981.0 * np.power(n, 4)
                    + 567.0 * np.power(n, 5)
                    + 269.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                (
                    -0.5925925925925926
                    * (
                        -960.0
                        - 1304.0 * n
                        + 3640.0 * np.power(n, 2)
                        + 5838.0 * np.power(n, 3)
                        + 1254.0 * np.power(n, 4)
                        - 5032.0 * np.power(n, 5)
                        - 4499.0 * np.power(n, 6)
                        - 1445.0 * np.power(n, 7)
                        + 285.0 * np.power(n, 8)
                        + 503.0 * np.power(n, 9)
                        + 136.0 * np.power(n, 10)
                    )
                )
                / (
                    (-2.0 + n)
                    * np.power(-1.0 + n, 2)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 2)
                )
                + (
                    1.1851851851851851
                    * (
                        36.0
                        + 137.0 * n
                        + 210.0 * np.power(n, 2)
                        + 95.0 * np.power(n, 3)
                        + 52.0 * np.power(n, 4)
                    )
                    * S1
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2))
            )
            * Sm2
            - (
                1.1851851851851851
                * (
                    36.0
                    + 85.0 * n
                    + 126.0 * np.power(n, 2)
                    + 49.0 * np.power(n, 3)
                    + 26.0 * np.power(n, 4)
                )
                * Sm21
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2))
            + (
                0.5925925925925926
                * (
                    -192.0
                    - 334.0 * n
                    - 373.0 * np.power(n, 2)
                    + 253.0 * np.power(n, 3)
                    + 291.0 * np.power(n, 4)
                    + 159.0 * np.power(n, 5)
                    + 52.0 * np.power(n, 6)
                )
                * Sm3
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    122.70623136505614
                    - 0.37037037037037035 * np.power(S1, 4)
                    - 7.555555555555555 * np.power(S1, 2) * S2
                    - 13.555555555555555 * np.power(S2, 2)
                    + 16.0 * S211
                    + (64.0 * S2m1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + 10.666666666666666 * S31
                    - 39.55555555555556 * S4
                    - (64.0 * S2 * Sm1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + (
                        -19.555555555555557 * np.power(S1, 2)
                        - 19.555555555555557 * S2
                        + (64.0 * Sm1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    )
                    * Sm2
                    - 3.5555555555555554 * np.power(Sm2, 2)
                    + S1
                    * (
                        -17.095920400492005
                        - 7.111111111111111 * S21
                        - 27.85185185185185 * S3
                        + 32.0 * Sm21
                    )
                    - 39.111111111111114 * Sm211
                    + 24.88888888888889 * Sm22
                    - (64.0 * Sm2m1) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    - 33.77777777777778 * S1 * Sm3
                    + 30.22222222222222 * Sm31
                    - 35.55555555555556 * Sm4
                )
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
    )
    a_gq_l3 = (
        -1.0
        * (2.0 + n + np.power(n, 2))
        * (
            1.1851851851851851 * (2.0 + nf)
            + 0.8888888888888888
            * (
                (
                    0.4444444444444444
                    * (
                        48.0
                        + 44.0 * n
                        + 52.0 * np.power(n, 2)
                        + 19.0 * np.power(n, 3)
                        + 17.0 * np.power(n, 4)
                        + 9.0 * np.power(n, 5)
                        + 3.0 * np.power(n, 6)
                    )
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
                - 1.7777777777777777 * S1
            )
            + 2.0
            * (
                (
                    -0.8888888888888888
                    * (
                        4.0
                        - 18.0 * n
                        - 7.0 * np.power(n, 2)
                        + 22.0 * np.power(n, 3)
                        + 11.0 * np.power(n, 4)
                    )
                )
                / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                + 1.7777777777777777 * S1
            )
        )
    ) / ((-1.0 + n) * n * (1.0 + n))
    a_gq_l2 = (
        0.3333333333333333
        * (
            (
                3.5555555555555554
                * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            - (10.666666666666666 * (2.0 + n + np.power(n, 2)) * S1)
            / ((-1.0 + n) * n * (1.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                -0.2222222222222222
                * (
                    -768.0
                    + 112.0 * n
                    - 720.0 * np.power(n, 2)
                    - 1616.0 * np.power(n, 3)
                    - 3236.0 * np.power(n, 4)
                    - 2451.0 * np.power(n, 5)
                    - 526.0 * np.power(n, 6)
                    + 604.0 * np.power(n, 7)
                    + 450.0 * np.power(n, 8)
                    + 87.0 * np.power(n, 9)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                1.7777777777777777
                * (-6.0 + 17.0 * n + 4.0 * np.power(n, 2) + 7.0 * np.power(n, 3))
                * S1
            )
            / ((-1.0 + n) * np.power(n, 2) * (1.0 + n))
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (np.power(S1, 2) - 5.0 * S2)
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
        + 2.0
        * (
            (
                0.4444444444444444
                * (
                    384.0
                    + 896.0 * n
                    + 528.0 * np.power(n, 2)
                    - 124.0 * np.power(n, 3)
                    - 820.0 * np.power(n, 4)
                    - 309.0 * np.power(n, 5)
                    + 694.0 * np.power(n, 6)
                    + 860.0 * np.power(n, 7)
                    + 414.0 * np.power(n, 8)
                    + 69.0 * np.power(n, 9)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                3.5555555555555554
                * (
                    30.0
                    - 7.0 * n
                    - 33.0 * np.power(n, 2)
                    - 8.0 * np.power(n, 3)
                    - 1.0 * np.power(n, 4)
                    + np.power(n, 5)
                )
                * S1
            )
            / (np.power(-1.0 + n, 2) * np.power(n, 2) * (1.0 + n) * (2.0 + n))
            + (
                (2.0 + n + np.power(n, 2))
                * (-2.6666666666666665 * np.power(S1, 2) - 8.0 * S2 - 16.0 * Sm2)
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
    )
    a_gq_l1 = (
        (-12.246913580246913 * (2.0 + n + np.power(n, 2)))
        / ((-1.0 + n) * n * (1.0 + n))
        - 0.3333333333333333
        * nf
        * (
            (
                1.1851851851851851
                * (
                    38.0
                    + 80.0 * n
                    + 86.0 * np.power(n, 2)
                    + 81.0 * np.power(n, 3)
                    + 19.0 * np.power(n, 4)
                )
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3))
            + (
                3.5555555555555554
                * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                * S1
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            + (
                (2.0 + n + np.power(n, 2))
                * (-5.333333333333333 * np.power(S1, 2) - 5.333333333333333 * S2)
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
        - 0.8888888888888888
        * (
            (
                -0.07407407407407407
                * (
                    3456.0
                    + 9504.0 * n
                    + 18240.0 * np.power(n, 2)
                    + 65344.0 * np.power(n, 3)
                    + 81160.0 * np.power(n, 4)
                    + 44386.0 * np.power(n, 5)
                    + 3704.0 * np.power(n, 6)
                    - 32981.0 * np.power(n, 7)
                    - 31663.0 * np.power(n, 8)
                    - 11406.0 * np.power(n, 9)
                    + 436.0 * np.power(n, 10)
                    + 1545.0 * np.power(n, 11)
                    + 339.0 * np.power(n, 12)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 2)
            )
            + (
                0.2962962962962963
                * (
                    -72.0
                    - 408.0 * n
                    - 446.0 * np.power(n, 2)
                    + 175.0 * np.power(n, 3)
                    + 472.0 * np.power(n, 4)
                    + 243.0 * np.power(n, 5)
                    + 164.0 * np.power(n, 6)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3))
            - (
                0.4444444444444444
                * (
                    -24.0
                    + 74.0 * n
                    + 135.0 * np.power(n, 2)
                    + 68.0 * np.power(n, 3)
                    + 43.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                0.4444444444444444
                * (
                    -288.0
                    - 480.0 * n
                    - 1444.0 * np.power(n, 2)
                    - 2036.0 * np.power(n, 3)
                    - 807.0 * np.power(n, 4)
                    + 802.0 * np.power(n, 5)
                    + 1056.0 * np.power(n, 6)
                    + 706.0 * np.power(n, 7)
                    + 187.0 * np.power(n, 8)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * (2.0 + n)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    -76.93164180221403
                    - 0.8888888888888888 * np.power(S1, 3)
                    + 29.333333333333332 * S1 * S2
                    - 10.666666666666666 * S21
                    + 19.555555555555557 * S3
                    - (128.0 * Sm2) / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                )
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
        - 2.0
        * (
            (
                0.2962962962962963
                * (
                    288.0
                    + 7440.0 * n
                    + 28928.0 * np.power(n, 2)
                    + 49136.0 * np.power(n, 3)
                    + 44966.0 * np.power(n, 4)
                    + 23063.0 * np.power(n, 5)
                    + 5814.0 * np.power(n, 6)
                    + 8033.0 * np.power(n, 7)
                    + 14891.0 * np.power(n, 8)
                    + 13239.0 * np.power(n, 9)
                    + 7232.0 * np.power(n, 10)
                    + 2301.0 * np.power(n, 11)
                    + 301.0 * np.power(n, 12)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                0.2962962962962963
                * (
                    576.0
                    - 1680.0 * n
                    - 8464.0 * np.power(n, 2)
                    - 7168.0 * np.power(n, 3)
                    + 4460.0 * np.power(n, 4)
                    + 11533.0 * np.power(n, 5)
                    + 9317.0 * np.power(n, 6)
                    + 5347.0 * np.power(n, 7)
                    + 2139.0 * np.power(n, 8)
                    + 356.0 * np.power(n, 9)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                0.4444444444444444
                * (
                    -264.0
                    - 88.0 * n
                    + 324.0 * np.power(n, 2)
                    + 127.0 * np.power(n, 3)
                    + 86.0 * np.power(n, 4)
                    + 31.0 * np.power(n, 5)
                )
                * np.power(S1, 2)
            )
            / (np.power(-1.0 + n, 2) * np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                1.3333333333333333
                * (
                    56.0
                    - 92.0 * np.power(n, 2)
                    - 1.0 * np.power(n, 3)
                    + 37.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 15.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            - (
                5.333333333333333
                * (
                    -40.0
                    - 72.0 * n
                    - 76.0 * np.power(n, 2)
                    + 39.0 * np.power(n, 3)
                    + 51.0 * np.power(n, 4)
                    + 37.0 * np.power(n, 5)
                    + 13.0 * np.power(n, 6)
                )
                * Sm2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    76.93164180221403
                    + 0.8888888888888888 * np.power(S1, 3)
                    + 18.666666666666668 * S1 * S2
                    + 28.444444444444443 * S3
                    + 42.666666666666664 * S1 * Sm2
                    - 21.333333333333332 * Sm21
                    + 21.333333333333332 * Sm3
                )
            )
            / ((-1.0 + n) * n * (1.0 + n))
        )
    )
    return a_gq_l0 + a_gq_l1 * L + a_gq_l2 * L**2 + a_gq_l3 * L**3
