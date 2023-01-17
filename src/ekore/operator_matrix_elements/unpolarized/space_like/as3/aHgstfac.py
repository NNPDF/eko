# pylint: skip-file
import numba as nb
import numpy as np


@nb.njit(cache=True)
def A_Hgstfac(n, sx, nf):
    r"""Computes the approximate incomplete part of :math:`A_{Hg}^{S,(3)}(N)`
    proportional to :math:`T_{F}`.
    The expression is presented in :cite:`Blumlein:2017wxd` (eq 3.1).

    When using the code, please cite the complete list of references
    available in :mod:`ekore.matching_conditions.as3`.

    The expression contains some unknown parts which are set to 0.
    However we have included a shift to impose the gluon
    momentum conservation for both the parts proportional to
    :math:`n_{f}^{0,1}` independently.
    In oder to do a minimal corrections we have added terms which
    can modify only the sub-leading terms for N to 0,1,infinity.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex

    """
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    S3, S21, _, Sm21, _, Sm3 = sx[2]
    S4, S31, S211, Sm22, Sm211, Sm31, Sm4 = sx[3]
    momentum_conservation_shift = (
        -136.47358087568801 / n**3 * nf - 375.88217393160176 / n**2
    )
    return momentum_conservation_shift + (
        (-1.0684950250307503 * (2.0 + n + np.power(n, 2))) / (n * (1.0 + n) * (2.0 + n))
        + 1.3333333333333333
        * (
            0.25
            * (
                1.6449340668482262
                * (
                    (
                        0.2222222222222222
                        * (
                            1728.0
                            + 4992.0 * n
                            + 8944.0 * np.power(n, 2)
                            + 11680.0 * np.power(n, 3)
                            + 4444.0 * np.power(n, 4)
                            - 4900.0 * np.power(n, 5)
                            - 6377.0 * np.power(n, 6)
                            + 617.0 * np.power(n, 7)
                            + 6930.0 * np.power(n, 8)
                            + 6142.0 * np.power(n, 9)
                            + 2331.0 * np.power(n, 10)
                            + 333.0 * np.power(n, 11)
                        )
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 4)
                        * np.power(1.0 + n, 4)
                        * np.power(2.0 + n, 3)
                    )
                    + (
                        8.88888888888889
                        * (6.0 + 11.0 * n + 4.0 * np.power(n, 2) + np.power(n, 3))
                        * S1
                    )
                    / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
                )
                + (
                    1.6449340668482262
                    * (2.0 + n + np.power(n, 2))
                    * (-13.333333333333334 * np.power(S1, 2) + 8.0 * S2)
                )
                / (n * (1.0 + n) * (2.0 + n))
                + nf
                * (
                    (
                        0.00411522633744856
                        * (
                            -1.24416e6
                            - 7.865856e6 * n
                            - 2.3256576e7 * np.power(n, 2)
                            - 4.2534912e7 * np.power(n, 3)
                            - 5.3947712e7 * np.power(n, 4)
                            - 5.5711424e7 * np.power(n, 5)
                            - 4.075048e7 * np.power(n, 6)
                            - 1.0343664e7 * np.power(n, 7)
                            + 1.264032e7 * np.power(n, 8)
                            + 1.1884298e7 * np.power(n, 9)
                            - 2.970289e6 * np.power(n, 10)
                            - 1.0465411e7 * np.power(n, 11)
                            - 5.568833e6 * np.power(n, 12)
                            + 575913.0 * np.power(n, 13)
                            + 1.874085e6 * np.power(n, 14)
                            + 879391.0 * np.power(n, 15)
                            + 186525.0 * np.power(n, 16)
                            + 15777.0 * np.power(n, 17)
                        )
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 6)
                        * np.power(1.0 + n, 6)
                        * np.power(2.0 + n, 5)
                    )
                    - (
                        0.3950617283950617
                        * (
                            141.0
                            + 521.0 * n
                            + 789.0 * np.power(n, 2)
                            + 185.0 * np.power(n, 3)
                            + 10.0 * np.power(n, 4)
                        )
                        * np.power(S1, 2)
                    )
                    / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
                    + (
                        0.3950617283950617
                        * (
                            24.0
                            + 83.0 * n
                            + 49.0 * np.power(n, 2)
                            + 10.0 * np.power(n, 3)
                        )
                        * np.power(S1, 3)
                    )
                    / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
                    + 1.6449340668482262
                    * (
                        (
                            0.2222222222222222
                            * (-2.0 + n)
                            * (
                                864.0
                                + 3264.0 * n
                                + 6232.0 * np.power(n, 2)
                                + 9804.0 * np.power(n, 3)
                                + 10888.0 * np.power(n, 4)
                                + 9325.0 * np.power(n, 5)
                                + 6717.0 * np.power(n, 6)
                                + 3842.0 * np.power(n, 7)
                                + 1606.0 * np.power(n, 8)
                                + 405.0 * np.power(n, 9)
                                + 45.0 * np.power(n, 10)
                            )
                        )
                        / (
                            (-1.0 + n)
                            * np.power(n, 4)
                            * np.power(1.0 + n, 4)
                            * np.power(2.0 + n, 3)
                        )
                        + (
                            1.7777777777777777
                            * (
                                12.0
                                + 28.0 * n
                                + 11.0 * np.power(n, 2)
                                + 5.0 * np.power(n, 3)
                            )
                            * S1
                        )
                        / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
                    )
                    + (
                        0.2962962962962963
                        * (
                            -5184.0
                            - 16992.0 * n
                            - 27808.0 * np.power(n, 2)
                            - 39024.0 * np.power(n, 3)
                            - 31384.0 * np.power(n, 4)
                            - 19422.0 * np.power(n, 5)
                            - 13965.0 * np.power(n, 6)
                            - 6819.0 * np.power(n, 7)
                            - 398.0 * np.power(n, 8)
                            + 1416.0 * np.power(n, 9)
                            + 547.0 * np.power(n, 10)
                            + 57.0 * np.power(n, 11)
                        )
                        * S2
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 4)
                        * np.power(1.0 + n, 4)
                        * np.power(2.0 + n, 3)
                    )
                    + S1
                    * (
                        (
                            -0.06584362139917696
                            * (
                                -2670.0
                                - 10217.0 * n
                                - 7454.0 * np.power(n, 2)
                                - 5165.0 * np.power(n, 3)
                                - 924.0 * np.power(n, 4)
                                + 230.0 * np.power(n, 5)
                            )
                        )
                        / (np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
                        + (
                            1.1851851851851851
                            * (
                                24.0
                                + 83.0 * n
                                + 49.0 * np.power(n, 2)
                                + 10.0 * np.power(n, 3)
                            )
                            * S2
                        )
                        / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
                    )
                    - (42.666666666666664 * (-2.0 - 3.0 * n + np.power(n, 2)) * S21)
                    / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
                    - (
                        0.19753086419753085
                        * (
                            3888.0
                            + 5376.0 * n
                            + 6832.0 * np.power(n, 2)
                            + 7472.0 * np.power(n, 3)
                            + 9129.0 * np.power(n, 4)
                            + 1736.0 * np.power(n, 5)
                            - 2382.0 * np.power(n, 6)
                            - 976.0 * np.power(n, 7)
                            + 29.0 * np.power(n, 8)
                        )
                        * S3
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 3)
                        * np.power(1.0 + n, 3)
                        * np.power(2.0 + n, 2)
                    )
                    + (
                        (2.0 + n + np.power(n, 2))
                        * (
                            -8.772981689857207 * np.power(S1, 2)
                            - 1.1851851851851851 * np.power(S1, 4)
                            + 1.2020569031595942
                            * (
                                (
                                    -6.222222222222222
                                    * (
                                        -24.0
                                        - 28.0 * n
                                        - 38.0 * np.power(n, 2)
                                        - 17.0 * np.power(n, 3)
                                        - 1.0 * np.power(n, 4)
                                        + 9.0 * np.power(n, 5)
                                        + 3.0 * np.power(n, 6)
                                    )
                                )
                                / (
                                    (-1.0 + n)
                                    * np.power(n, 2)
                                    * np.power(1.0 + n, 2)
                                    * (2.0 + n)
                                )
                                + 24.88888888888889 * S1
                            )
                            - 7.111111111111111 * np.power(S1, 2) * S2
                            - 14.222222222222221 * np.power(S2, 2)
                            + 85.33333333333333 * S211
                            + S1 * (-42.666666666666664 * S21 - 9.481481481481481 * S3)
                            - 42.666666666666664 * S31
                            + 28.444444444444443 * S4
                        )
                    )
                    / (n * (1.0 + n) * (2.0 + n))
                )
            )
            + 1.5
            * (
                (
                    -212.26414844076453
                    * (
                        -16.0
                        - 14.0 * np.power(n, 2)
                        - 25.0 * np.power(n, 3)
                        - 5.0 * np.power(n, 4)
                        + 9.0 * np.power(n, 5)
                        + 3.0 * np.power(n, 6)
                    )
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * np.power(2.0 + n, 2)
                )
                + 1.6449340668482262
                * (
                    (
                        0.05555555555555555
                        * (
                            3552.0
                            + 17200.0 * n
                            + 46032.0 * np.power(n, 2)
                            + 76456.0 * np.power(n, 3)
                            + 88078.0 * np.power(n, 4)
                            + 65115.0 * np.power(n, 5)
                            + 27752.0 * np.power(n, 6)
                            + 2506.0 * np.power(n, 7)
                            - 1566.0 * np.power(n, 8)
                            - 261.0 * np.power(n, 9)
                        )
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 3)
                        * np.power(1.0 + n, 3)
                        * np.power(2.0 + n, 3)
                    )
                    - (
                        0.4444444444444444
                        * (
                            288.0
                            + 48.0 * n
                            - 3392.0 * np.power(n, 2)
                            - 5768.0 * np.power(n, 3)
                            - 3602.0 * np.power(n, 4)
                            + 1523.0 * np.power(n, 5)
                            + 5338.0 * np.power(n, 6)
                            + 4868.0 * np.power(n, 7)
                            + 2088.0 * np.power(n, 8)
                            + 337.0 * np.power(n, 9)
                        )
                        * S1
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 3)
                        * np.power(1.0 + n, 3)
                        * np.power(2.0 + n, 3)
                    )
                    + (
                        2.6666666666666665
                        * (
                            36.0
                            + 56.0 * n
                            + 29.0 * np.power(n, 2)
                            - 137.0 * np.power(n, 3)
                            - 120.0 * np.power(n, 4)
                            - 9.0 * np.power(n, 5)
                            + np.power(n, 6)
                        )
                        * np.power(S1, 2)
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 2)
                        * np.power(1.0 + n, 2)
                        * np.power(2.0 + n, 2)
                    )
                )
                + (
                    (2.0 + n + np.power(n, 2))
                    * (
                        -212.26414844076453 * S1
                        + 1.6449340668482262
                        * (
                            32.0 * np.power(S1, 3)
                            - (
                                12.0
                                * (
                                    -4.0
                                    - 4.0 * n
                                    - 3.0 * np.power(n, 2)
                                    + 2.0 * np.power(n, 3)
                                    + np.power(n, 4)
                                )
                                * S2
                            )
                            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                            - 8.0 * S3
                            + (
                                (-8.0 * (1.0 + 3.0 * n + 3.0 * np.power(n, 2)))
                                / (n * (1.0 + n))
                                + 16.0 * S1
                            )
                            * Sm2
                            + 16.0 * Sm21
                            - 8.0 * Sm3
                        )
                    )
                )
                / (n * (1.0 + n) * (2.0 + n))
            )
        )
        + 4.5
        * (
            (
                77.92727282720195
                * (-2.0 + n)
                * (3.0 + n)
                * (
                    4.0
                    + 4.0 * n
                    + 7.0 * np.power(n, 2)
                    + 6.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                7.05120034829508
                * (
                    -56.0
                    - 20.0 * n
                    - 62.0 * np.power(n, 2)
                    - 75.0 * np.power(n, 3)
                    - 15.0 * np.power(n, 4)
                    + 27.0 * np.power(n, 5)
                    + 9.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + 1.6449340668482262
            * (
                (
                    0.2222222222222222
                    * (
                        -3456.0
                        - 17184.0 * n
                        - 39184.0 * np.power(n, 2)
                        - 57200.0 * np.power(n, 3)
                        - 54000.0 * np.power(n, 4)
                        - 36634.0 * np.power(n, 5)
                        - 19177.0 * np.power(n, 6)
                        - 16952.0 * np.power(n, 7)
                        - 17658.0 * np.power(n, 8)
                        - 8937.0 * np.power(n, 9)
                        - 997.0 * np.power(n, 10)
                        + 1190.0 * np.power(n, 11)
                        + 552.0 * np.power(n, 12)
                        + 69.0 * np.power(n, 13)
                    )
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 4)
                    * np.power(1.0 + n, 4)
                    * np.power(2.0 + n, 4)
                )
                + (
                    0.4444444444444444
                    * (
                        -864.0
                        - 2160.0 * n
                        + 680.0 * np.power(n, 2)
                        + 2820.0 * np.power(n, 3)
                        + 3078.0 * np.power(n, 4)
                        - 601.0 * np.power(n, 5)
                        - 809.0 * np.power(n, 6)
                        + 1298.0 * np.power(n, 7)
                        + 1124.0 * np.power(n, 8)
                        + 515.0 * np.power(n, 9)
                        + 103.0 * np.power(n, 10)
                    )
                    * S1
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 3)
                )
                - (
                    1.3333333333333333
                    * (
                        -48.0
                        - 116.0 * n
                        + 4.0 * np.power(n, 2)
                        - 85.0 * np.power(n, 3)
                        - 87.0 * np.power(n, 4)
                        + 33.0 * np.power(n, 5)
                        + 11.0 * np.power(n, 6)
                    )
                    * np.power(S1, 2)
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * np.power(2.0 + n, 2)
                )
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    184.0593470475842 * S1
                    + 1.6449340668482262
                    * (
                        -16.0 * np.power(S1, 3)
                        - (
                            1.3333333333333333
                            * (
                                -48.0
                                - 70.0 * n
                                - 59.0 * np.power(n, 2)
                                + 22.0 * np.power(n, 3)
                                + 11.0 * np.power(n, 4)
                            )
                            * S2
                        )
                        / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                        - 32.0 * S1 * S2
                        - 8.0 * S3
                        + (
                            (
                                -2.6666666666666665
                                * (
                                    -36.0
                                    - 58.0 * n
                                    - 47.0 * np.power(n, 2)
                                    + 22.0 * np.power(n, 3)
                                    + 11.0 * np.power(n, 4)
                                )
                            )
                            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                            - 48.0 * S1
                        )
                        * Sm2
                        + 16.0 * Sm21
                        - 8.0 * Sm3
                    )
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (106.13207422038226 * (-1.0 + n) * (-2.0 + 3.0 * n + 3.0 * np.power(n, 2)))
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            + 1.6449340668482262
            * (
                (
                    0.5
                    * (
                        48.0
                        + 184.0 * n
                        + 176.0 * np.power(n, 2)
                        - 222.0 * np.power(n, 3)
                        - 947.0 * np.power(n, 4)
                        - 1374.0 * np.power(n, 5)
                        - 1196.0 * np.power(n, 6)
                        - 612.0 * np.power(n, 7)
                        - 153.0 * np.power(n, 8)
                    )
                )
                / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
                + (
                    8.0
                    * (
                        -10.0
                        - 29.0 * n
                        - 21.0 * np.power(n, 2)
                        + 8.0 * np.power(n, 3)
                        + 39.0 * np.power(n, 4)
                        + 36.0 * np.power(n, 5)
                        + 13.0 * np.power(n, 6)
                    )
                    * S1
                )
                / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
                + (
                    4.0
                    * (
                        20.0
                        + 48.0 * n
                        + 43.0 * np.power(n, 2)
                        + 14.0 * np.power(n, 3)
                        + 3.0 * np.power(n, 4)
                    )
                    * np.power(S1, 2)
                )
                / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            )
            + (
                1.6449340668482262
                * (2.0 + n + np.power(n, 2))
                * (
                    -16.0 * np.power(S1, 3)
                    - (8.0 * (2.0 + 3.0 * n + 3.0 * np.power(n, 2)) * S2)
                    / (n * (1.0 + n))
                    + 32.0 * S1 * S2
                    + 16.0 * S3
                    + (-16.0 / (n * (1.0 + n)) + 32.0 * S1) * Sm2
                    - 32.0 * Sm21
                    + 16.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * (
            1.6449340668482262
            * (
                (
                    -0.4444444444444444
                    * (
                        672.0
                        + 3008.0 * n
                        + 5352.0 * np.power(n, 2)
                        + 6500.0 * np.power(n, 3)
                        + 5180.0 * np.power(n, 4)
                        + 3171.0 * np.power(n, 5)
                        + 2134.0 * np.power(n, 6)
                        + 1148.0 * np.power(n, 7)
                        + 414.0 * np.power(n, 8)
                        + 69.0 * np.power(n, 9)
                    )
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 3)
                )
                + (
                    17.77777777777778
                    * (
                        4.0
                        - 1.0 * n
                        + np.power(n, 2)
                        + 4.0 * np.power(n, 3)
                        + np.power(n, 4)
                    )
                    * S1
                )
                / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            )
            + (
                1.6449340668482262
                * (2.0 + n + np.power(n, 2))
                * (
                    13.333333333333334 * np.power(S1, 2)
                    + 13.333333333333334 * S2
                    + 26.666666666666668 * Sm2
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
            + nf
            * (
                (
                    -0.03292181069958848
                    * (
                        3456.0
                        + 18432.0 * n
                        + 33504.0 * np.power(n, 2)
                        - 22912.0 * np.power(n, 3)
                        - 281016.0 * np.power(n, 4)
                        - 465872.0 * np.power(n, 5)
                        - 806374.0 * np.power(n, 6)
                        - 1.459136e6 * np.power(n, 7)
                        - 1.48494e6 * np.power(n, 8)
                        - 377441.0 * np.power(n, 9)
                        + 849246.0 * np.power(n, 10)
                        + 1.139033e6 * np.power(n, 11)
                        + 692290.0 * np.power(n, 12)
                        + 237011.0 * np.power(n, 13)
                        + 44514.0 * np.power(n, 14)
                        + 3597.0 * np.power(n, 15)
                    )
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 5)
                    * np.power(1.0 + n, 5)
                    * np.power(2.0 + n, 5)
                )
                + (
                    0.09876543209876543
                    * (
                        1256.0
                        + 3172.0 * n
                        + 6816.0 * np.power(n, 2)
                        + 6430.0 * np.power(n, 3)
                        + 2355.0 * np.power(n, 4)
                        + 271.0 * np.power(n, 5)
                        + 22.0 * np.power(n, 6)
                    )
                    * np.power(S1, 2)
                )
                / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
                - (
                    0.19753086419753085
                    * (
                        134.0
                        + 439.0 * n
                        + 344.0 * np.power(n, 2)
                        + 107.0 * np.power(n, 3)
                        + 20.0 * np.power(n, 4)
                    )
                    * np.power(S1, 3)
                )
                / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
                + 1.6449340668482262
                * (
                    (
                        -0.4444444444444444
                        * (
                            96.0
                            + 224.0 * n
                            - 48.0 * np.power(n, 2)
                            - 244.0 * np.power(n, 3)
                            - 610.0 * np.power(n, 4)
                            - 501.0 * np.power(n, 5)
                            - 32.0 * np.power(n, 6)
                            + 146.0 * np.power(n, 7)
                            + 90.0 * np.power(n, 8)
                            + 15.0 * np.power(n, 9)
                        )
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 3)
                        * np.power(1.0 + n, 3)
                        * np.power(2.0 + n, 3)
                    )
                    - (
                        1.7777777777777777
                        * (
                            20.0
                            + 76.0 * n
                            + 59.0 * np.power(n, 2)
                            + 20.0 * np.power(n, 3)
                            + 5.0 * np.power(n, 4)
                        )
                        * S1
                    )
                    / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
                )
                + (
                    0.09876543209876543
                    * (
                        -1728.0
                        - 4032.0 * n
                        - 3128.0 * np.power(n, 2)
                        - 6644.0 * np.power(n, 3)
                        + 7720.0 * np.power(n, 4)
                        + 15770.0 * np.power(n, 5)
                        + 6901.0 * np.power(n, 6)
                        + 806.0 * np.power(n, 7)
                        - 117.0 * np.power(n, 8)
                        + 4.0 * np.power(n, 9)
                    )
                    * S2
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 3)
                )
                + S1
                * (
                    (
                        0.06584362139917696
                        * (
                            864.0
                            - 2672.0 * n
                            - 11408.0 * np.power(n, 2)
                            - 73764.0 * np.power(n, 3)
                            - 73982.0 * np.power(n, 4)
                            + 29418.0 * np.power(n, 5)
                            + 87216.0 * np.power(n, 6)
                            + 61598.0 * np.power(n, 7)
                            + 23603.0 * np.power(n, 8)
                            + 5292.0 * np.power(n, 9)
                            + 491.0 * np.power(n, 10)
                        )
                    )
                    / (
                        (-1.0 + n)
                        * np.power(n, 2)
                        * np.power(1.0 + n, 4)
                        * np.power(2.0 + n, 4)
                    )
                    - (
                        0.5925925925925926
                        * (
                            214.0
                            + 779.0 * n
                            + 544.0 * np.power(n, 2)
                            + 151.0 * np.power(n, 3)
                            + 40.0 * np.power(n, 4)
                        )
                        * S2
                    )
                    / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
                )
                - (
                    2.3703703703703702
                    * (
                        20.0
                        + 85.0 * n
                        + 50.0 * np.power(n, 2)
                        + 11.0 * np.power(n, 3)
                        + 5.0 * np.power(n, 4)
                    )
                    * S21
                )
                / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
                - (
                    0.3950617283950617
                    * (
                        648.0
                        + 496.0 * n
                        + 370.0 * np.power(n, 2)
                        + 725.0 * np.power(n, 3)
                        + 1155.0 * np.power(n, 4)
                        + 429.0 * np.power(n, 5)
                        + 65.0 * np.power(n, 6)
                    )
                    * S3
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * np.power(2.0 + n, 2)
                )
                + (
                    0.3950617283950617
                    * (
                        448.0
                        + 284.0 * n
                        + 1794.0 * np.power(n, 2)
                        + 2552.0 * np.power(n, 3)
                        + 1257.0 * np.power(n, 4)
                        + 278.0 * np.power(n, 5)
                        + 47.0 * np.power(n, 6)
                    )
                    * Sm2
                )
                / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
                + (
                    1.1851851851851851
                    * (
                        216.0
                        - 20.0 * n
                        - 548.0 * np.power(n, 2)
                        - 511.0 * np.power(n, 3)
                        - 339.0 * np.power(n, 4)
                        - 99.0 * np.power(n, 5)
                        + 5.0 * np.power(n, 6)
                    )
                    * Sm3
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * np.power(2.0 + n, 2)
                )
                - (
                    7.111111111111111
                    * (
                        36.0
                        - 20.0 * n
                        - 143.0 * np.power(n, 2)
                        - 61.0 * np.power(n, 3)
                        - 24.0 * np.power(n, 4)
                        - 9.0 * np.power(n, 5)
                        + 5.0 * np.power(n, 6)
                    )
                    * (S1 * Sm2 - 1.0 * Sm21 + Sm3)
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 2)
                    * np.power(1.0 + n, 2)
                    * np.power(2.0 + n, 2)
                )
                + (
                    (2.0 + n + np.power(n, 2))
                    * (
                        1.2020569031595942
                        * (
                            (49.77777777777778 * (1.0 + n + np.power(n, 2)))
                            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                            - 24.88888888888889 * S1
                        )
                        + 1.1851851851851851 * np.power(S1, 4)
                        + 19.555555555555557 * np.power(S1, 2) * S2
                        + 8.88888888888889 * np.power(S2, 2)
                        - 46.22222222222222 * S211
                        + S1 * (24.88888888888889 * S21 + 69.92592592592592 * S3)
                        - 3.5555555555555554 * S31
                        + 71.11111111111111 * S4
                        + (
                            (-64.0 * (-1.0 + 2.0 * n) * S1) / ((-1.0 + n) * n)
                            + 42.666666666666664 * S2
                        )
                        * Sm2
                        + 1.6449340668482262
                        * (
                            5.333333333333333 * np.power(S1, 2)
                            + 5.333333333333333 * S2
                            + 10.666666666666666 * Sm2
                        )
                        + (64.0 * (-1.0 + 2.0 * n) * Sm21) / ((-1.0 + n) * n)
                        + 7.111111111111111 * Sm4
                        - 21.333333333333332 * (S2 * Sm2 - 1.0 * Sm22 + Sm4)
                        - 10.666666666666666 * (S1 * Sm3 - 1.0 * Sm31 + Sm4)
                        + 64.0
                        * (
                            S2 * Sm2
                            - 0.5 * (np.power(S1, 2) + S2) * Sm2
                            + Sm211
                            - 1.0 * Sm22
                            + S1 * (S1 * Sm2 - 1.0 * Sm21 + Sm3)
                            - 1.0 * Sm31
                            + Sm4
                        )
                    )
                )
                / (n * (1.0 + n) * (2.0 + n))
            )
        )
    )
