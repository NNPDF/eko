# pylint: disable=too-many-lines
import numba as nb
import numpy as np


@nb.njit(cache=True)
def A_Hq(n, sx, nf, L):  # pylint: disable=too-many-locals
    r"""Computes the |N3LO| singlet |OME| :math:`A_{Hq}^{S,(3)}(N)`.
    The expression is presented in :cite:`Ablinger_2015` (eq 5.1)
    and :cite:`Blumlein:2017wxd` (eq 3.1).

    When using the code, please cite the complete list of references
    available in :mod:`ekore.matching_conditions.as3`.

    The part proportional to :math:`n_f^0` includes non trivial weight-5
    harmonics and has been parametrized in Mellin space.
    For these pieces the accuracy wrt the exact expression is below
    0.001% (N<1000).
    All the other contributions are provided exact.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache
    nf : int
        number of active flavor below the threshold
    L : float
        :math:`\ln(\mu_F^2 / m_h^2)`

    Returns
    -------
    complex
        :math:`A_{Hq}^{S,(3)}(N)`

    """
    S1, _ = sx[0]
    S2, Sm2 = sx[1]
    S3, S21, _, Sm21, _, Sm3 = sx[2]
    S4, S31, S211, Sm22, Sm211, Sm31, Sm4 = sx[3]
    S5, _ = sx[4]

    # fit of:
    #  2^-N * ( H1 + 8.41439832211716)
    # with_
    #   H1 = S111l211 + S12l21 - S21l21 - S3l2
    H1fit = -(
        11.14288617196527 / n**6
        - 55.57776925405718 / n**5
        + 130.96786052283326 / n**4
        + 326.9731754784073 / n**3
        + 130.8337793228711 / n**2
        + 38.323678403287424 / n
        + 0.33551656325214146 * S1
        - (315.5404867667502 * S1) / n**4
        - (194.80677104863983 * S1) / n**3
        - (41.34701980451916 * S1) / n**2
        - (6.84440479473815 * S1) / n
        - 0.04638030687417067 * S1**2
        + 0.0029603058349121163 * S1**3
        - 0.00007344490079458677 * S1**4
        + 24.052937205797793 * S2
        + 79.59744541202188 * S3
        + 21.26650914027558 * S4
        - 153.5410422671263 * S5
    )

    # fit of:
    #  H2fit = prefactor * (-32.0 * H2 + 269.261 * S1l05)
    #  H3fit = prefactor * (64.0 * H2 - 538.521 * S1l05)
    #
    # with:
    #  prefactor = (2.0 + n + np.power(n, 2)) ** 2 / (
    #     (-1 + n) * (1 + n) ** 2 * n ** 2 * (2 + n)
    #  H26 = S211l2051 + S211l2105 - S22l205
    #  H27 = + S1111l21105 + S112l2051 - S112l2105  + S121l2105 - S13l205
    #  H2 = (-H26 + H27 + S1111l20511 + S1111l21051 - S121l2051 - S31l205) - S1l05 * H1 )
    H2fit = (
        1.0
        / (n - 1.0)
        * (
            210.28504428179983 / n**6
            - 1187.8628810061025 / n**5
            + 3272.309401484616 / n**4
            + 7910.0387388902745 / n**3
            + 302.1098733538397 / n**2
            - 271.6704798741525 / n
            - 0.1811765267875728 * S1
            - (8398.040966254019 * S1) / n**4
            - (1525.0229384552197 * S1) / n**3
            - (233.6634880221311 * S1) / n**2
            - (1.5773749318510508 * S1) / n
            + 0.027413160972699507 * S1**2
            - 0.0019046488424798599 * S1**3
            + 0.000051143314788330664 * S1**4
            - 530.1180895638906 * S2
            + 45.73696014976202 * S3
            + 4596.830317621372 * S4
            - 4009.6913365977616 * S5
        )
    )
    H3fit = (
        1.0
        / (n - 1.0)
        * (
            -420.57008856359965 / n**6
            + 2375.725762012205 / n**5
            - 6544.618802969232 / n**4
            - 15820.077477780549 / n**3
            - 604.2197467076794 / n**2
            + 543.340959748305 / n
            + 0.3623530535751456 * S1
            + (16796.081932508037 * S1) / n**4
            + (3050.0458769104393 * S1) / n**3
            + (467.3269760442622 * S1) / n**2
            + (3.1547498637021016 * S1) / n
            - 0.05482632194539901 * S1**2
            + 0.0038092976849597197 * S1**3
            - 0.00010228662957666133 * S1**4
            + 1060.2361791277813 * S2
            - 91.47392029952404 * S3
            - 9193.660635242744 * S4
            + 8019.382673195523 * S5
        )
    )
    a_Hq_l0 = (
        0.3333333333333333
        * nf
        * (
            (
                2.9243272299524024
                * (
                    -48.0
                    - 104.0 * n
                    - 56.0 * np.power(n, 2)
                    - 86.0 * np.power(n, 3)
                    - 11.0 * np.power(n, 4)
                    + 68.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                0.13168724279835392
                * (
                    -6912.0
                    - 35712.0 * n
                    - 77952.0 * np.power(n, 2)
                    - 84608.0 * np.power(n, 3)
                    - 24944.0 * np.power(n, 4)
                    - 12856.0 * np.power(n, 5)
                    - 8896.0 * np.power(n, 6)
                    + 59452.0 * np.power(n, 7)
                    + 89880.0 * np.power(n, 8)
                    + 56186.0 * np.power(n, 9)
                    + 23003.0 * np.power(n, 10)
                    + 7714.0 * np.power(n, 11)
                    + 1663.0 * np.power(n, 12)
                    + 158.0 * np.power(n, 13)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            - (
                0.3950617283950617
                * (
                    576.0
                    + 2112.0 * n
                    + 3040.0 * np.power(n, 2)
                    + 1648.0 * np.power(n, 3)
                    + 2244.0 * np.power(n, 4)
                    + 1848.0 * np.power(n, 5)
                    - 20.0 * np.power(n, 6)
                    + 30.0 * np.power(n, 7)
                    + 417.0 * np.power(n, 8)
                    + 176.0 * np.power(n, 9)
                    + 25.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                0.5925925925925926
                * (
                    -48.0
                    - 104.0 * n
                    - 56.0 * np.power(n, 2)
                    - 86.0 * np.power(n, 3)
                    - 11.0 * np.power(n, 4)
                    + 68.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                7.703703703703703
                * (
                    -48.0
                    - 104.0 * n
                    - 56.0 * np.power(n, 2)
                    - 86.0 * np.power(n, 3)
                    - 11.0 * np.power(n, 4)
                    + 68.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    29.917860700861013
                    - 8.772981689857207 * S1
                    - 0.5925925925925926 * np.power(S1, 3)
                    - 23.11111111111111 * S1 * S2
                    - 65.18518518518519 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                -5.848654459904805
                * (
                    24.0
                    + 124.0 * n
                    + 162.0 * np.power(n, 2)
                    + 193.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    + 29.0 * np.power(n, 5)
                    + 8.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                1.1851851851851851
                * (
                    (
                        -0.2222222222222222
                        * (
                            207360.0
                            + 1.026432e6 * n
                            + 2.192832e6 * np.power(n, 2)
                            + 3.109248e6 * np.power(n, 3)
                            + 4.514336e6 * np.power(n, 4)
                            + 8.472792e6 * np.power(n, 5)
                            + 1.2693884e7 * np.power(n, 6)
                            + 1.2958212e7 * np.power(n, 7)
                            + 9.333994e6 * np.power(n, 8)
                            + 4.877344e6 * np.power(n, 9)
                            + 1.87144e6 * np.power(n, 10)
                            + 559575.0 * np.power(n, 11)
                            + 145948.0 * np.power(n, 12)
                            + 32280.0 * np.power(n, 13)
                            + 4670.0 * np.power(n, 14)
                            + 293.0 * np.power(n, 15)
                        )
                    )
                    / (np.power(n, 5) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
                    + (
                        0.6666666666666666
                        * (
                            -17280.0
                            - 76896.0 * n
                            - 82368.0 * np.power(n, 2)
                            + 155864.0 * np.power(n, 3)
                            + 599060.0 * np.power(n, 4)
                            + 886552.0 * np.power(n, 5)
                            + 837697.0 * np.power(n, 6)
                            + 553796.0 * np.power(n, 7)
                            + 251778.0 * np.power(n, 8)
                            + 79990.0 * np.power(n, 9)
                            + 20431.0 * np.power(n, 10)
                            + 4658.0 * np.power(n, 11)
                            + 746.0 * np.power(n, 12)
                            + 52.0 * np.power(n, 13)
                        )
                        * S1
                    )
                    / (np.power(n, 4) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
                    - (
                        1.0
                        * (
                            -7200.0
                            - 3960.0 * n
                            + 22748.0 * np.power(n, 2)
                            + 37370.0 * np.power(n, 3)
                            + 40683.0 * np.power(n, 4)
                            + 34749.0 * np.power(n, 5)
                            + 18410.0 * np.power(n, 6)
                            + 5724.0 * np.power(n, 7)
                            + 1095.0 * np.power(n, 8)
                            + 133.0 * np.power(n, 9)
                            + 8.0 * np.power(n, 10)
                        )
                        * np.power(S1, 2)
                    )
                    / (np.power(n, 3) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
                    + (
                        (
                            7200.0
                            - 26280.0 * n
                            - 46100.0 * np.power(n, 2)
                            - 47454.0 * np.power(n, 3)
                            - 33693.0 * np.power(n, 4)
                            - 7014.0 * np.power(n, 5)
                            + 5392.0 * np.power(n, 6)
                            + 3284.0 * np.power(n, 7)
                            + 625.0 * np.power(n, 8)
                            + 40.0 * np.power(n, 9)
                        )
                        * S2
                    )
                    / (np.power(n, 3) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
                )
            )
            / ((-1.0 + n) * (3.0 + n) * (4.0 + n) * (5.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -136.76736320393604
                    + 17.545963379714415 * S1
                    + 1.1851851851851851 * np.power(S1, 3)
                    - 17.77777777777778 * S1 * S2
                    + 42.666666666666664 * S21
                    - 18.962962962962962 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                5.848654459904805
                * (
                    24.0
                    + 124.0 * n
                    + 162.0 * np.power(n, 2)
                    + 193.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    + 29.0 * np.power(n, 5)
                    + 8.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                0.3950617283950617
                * (
                    864.0
                    + 5616.0 * n
                    + 15984.0 * np.power(n, 2)
                    + 32344.0 * np.power(n, 3)
                    + 63406.0 * np.power(n, 4)
                    + 128195.0 * np.power(n, 5)
                    + 192416.0 * np.power(n, 6)
                    + 196942.0 * np.power(n, 7)
                    + 148026.0 * np.power(n, 8)
                    + 87182.0 * np.power(n, 9)
                    + 39593.0 * np.power(n, 10)
                    + 12793.0 * np.power(n, 11)
                    + 2599.0 * np.power(n, 12)
                    + 248.0 * np.power(n, 13)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            - (
                1.1851851851851851
                * (2.0 + n + np.power(n, 2))
                * (
                    86.0
                    + 230.0 * n
                    + 224.0 * np.power(n, 2)
                    + 105.0 * np.power(n, 3)
                    + 43.0 * np.power(n, 4)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                1.7777777777777777
                * (2.0 + n + np.power(n, 2))
                * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                1.7777777777777777
                * (
                    96.0
                    + 400.0 * n
                    + 628.0 * np.power(n, 2)
                    + 796.0 * np.power(n, 3)
                    + 565.0 * np.power(n, 4)
                    + 158.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    17.095920400492005
                    - 17.545963379714415 * S1
                    - 1.7777777777777777 * np.power(S1, 3)
                    - 5.333333333333333 * S1 * S2
                    + 17.77777777777778 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.3333333333333333
        * nf
        * (
            (
                -2.9243272299524024
                * (
                    -48.0
                    - 104.0 * n
                    - 56.0 * np.power(n, 2)
                    - 86.0 * np.power(n, 3)
                    - 11.0 * np.power(n, 4)
                    + 68.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                10.666666666666666
                * (
                    -32.0
                    - 208.0 * n
                    - 592.0 * np.power(n, 2)
                    - 904.0 * np.power(n, 3)
                    - 682.0 * np.power(n, 4)
                    - 473.0 * np.power(n, 5)
                    - 400.0 * np.power(n, 6)
                    + 38.0 * np.power(n, 7)
                    + 374.0 * np.power(n, 8)
                    + 252.0 * np.power(n, 9)
                    + 62.0 * np.power(n, 10)
                    + 5.0 * np.power(n, 11)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                21.333333333333332
                * (2.0 + 5.0 * n + np.power(n, 2))
                * (4.0 + 4.0 * n + 7.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (4.273980100123001 + 8.772981689857207 * S1 + 21.333333333333332 * S3)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                -1.6027425375461255
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                3.2898681336964524
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 28.0 * n
                    + 21.0 * np.power(n, 2)
                    + 106.0 * np.power(n, 3)
                    + 151.0 * np.power(n, 4)
                    + 108.0 * np.power(n, 5)
                    + 38.0 * np.power(n, 6)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                4.0
                * (
                    32.0
                    + 240.0 * n
                    + 496.0 * np.power(n, 2)
                    - 72.0 * np.power(n, 3)
                    - 1254.0 * np.power(n, 4)
                    + 339.0 * np.power(n, 5)
                    + 6106.0 * np.power(n, 6)
                    + 11692.0 * np.power(n, 7)
                    + 13272.0 * np.power(n, 8)
                    + 10762.0 * np.power(n, 9)
                    + 6049.0 * np.power(n, 10)
                    + 2139.0 * np.power(n, 11)
                    + 443.0 * np.power(n, 12)
                    + 56.0 * np.power(n, 13)
                    + 4.0 * np.power(n, 14)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 3)
            )
            + (
                6.579736267392905
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    - 10.0 * n
                    + np.power(n, 2)
                    + 4.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                8.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 44.0 * n
                    - 19.0 * np.power(n, 2)
                    - 11.0 * np.power(n, 3)
                    - 2.0 * np.power(n, 4)
                    + 2.0 * np.power(n, 5)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -4.0
                    - 18.0 * n
                    - 32.0 * np.power(n, 2)
                    - 5.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    - 20.0 * n
                    - 8.0 * np.power(n, 2)
                    + 56.0 * np.power(n, 3)
                    + 135.0 * np.power(n, 4)
                    + 102.0 * np.power(n, 5)
                    + 27.0 * np.power(n, 6)
                )
                * S2
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                (2.0 + 3.0 * n)
                * (2.0 + n + np.power(n, 2))
                * (-2.6666666666666665 * np.power(S1, 3) - 8.0 * S1 * S2)
            )
            / ((-1.0 + n) * np.power(n, 3) * (1.0 + n) * (2.0 + n))
            + (
                32.0
                * (-2.0 - 3.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * S21
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    - 22.0 * n
                    + 43.0 * np.power(n, 2)
                    + 48.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * S3
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    6.410970150184502 * S1
                    + 0.6666666666666666 * np.power(S1, 4)
                    + 1.6449340668482262 * (4.0 * np.power(S1, 2) - 12.0 * S2)
                    + 4.0 * np.power(S1, 2) * S2
                    + 2.0 * np.power(S2, 2)
                    - 64.0 * S211
                    + S1 * (32.0 * S21 + 5.333333333333333 * S3)
                    + 32.0 * S31
                    - 12.0 * S4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                1.6027425375461255
                * (
                    -1072.0
                    - 1008.0 * n
                    - 4120.0 * np.power(n, 2)
                    - 7320.0 * np.power(n, 3)
                    - 3299.0 * np.power(n, 4)
                    - 1487.0 * np.power(n, 5)
                    - 1089.0 * np.power(n, 6)
                    - 45.0 * np.power(n, 7)
                    + 192.0 * np.power(n, 8)
                    + 48.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                1.3333333333333333
                * (
                    -1728.0
                    - 13504.0 * n
                    - 45232.0 * np.power(n, 2)
                    - 83504.0 * np.power(n, 3)
                    - 88676.0 * np.power(n, 4)
                    - 48500.0 * np.power(n, 5)
                    + 9415.0 * np.power(n, 6)
                    + 50675.0 * np.power(n, 7)
                    + 57974.0 * np.power(n, 8)
                    + 41400.0 * np.power(n, 9)
                    + 25694.0 * np.power(n, 10)
                    + 18236.0 * np.power(n, 11)
                    + 11443.0 * np.power(n, 12)
                    + 4569.0 * np.power(n, 13)
                    + 978.0 * np.power(n, 14)
                    + 88.0 * np.power(n, 15)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 4)
            )
            - (
                1.3333333333333333
                * (
                    192.0
                    + 256.0 * n
                    + 176.0 * np.power(n, 2)
                    + 840.0 * np.power(n, 3)
                    + 944.0 * np.power(n, 4)
                    + 490.0 * np.power(n, 5)
                    + 662.0 * np.power(n, 6)
                    + 735.0 * np.power(n, 7)
                    + 363.0 * np.power(n, 8)
                    + 75.0 * np.power(n, 9)
                    + 3.0 * np.power(n, 10)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    (
                        -0.4444444444444444
                        * (
                            -8.0
                            - 10.0 * n
                            + np.power(n, 2)
                            + 4.0 * np.power(n, 3)
                            + 5.0 * np.power(n, 4)
                        )
                        * np.power(S1, 3)
                    )
                    / (n * (1.0 + n))
                    + 1.6449340668482262
                    * (
                        (
                            2.0
                            * (
                                -12.0
                                - 28.0 * n
                                + 21.0 * np.power(n, 2)
                                + 106.0 * np.power(n, 3)
                                + 151.0 * np.power(n, 4)
                                + 108.0 * np.power(n, 5)
                                + 38.0 * np.power(n, 6)
                            )
                        )
                        / (np.power(n, 2) * np.power(1.0 + n, 2))
                        - (
                            4.0
                            * (
                                -8.0
                                - 10.0 * n
                                + np.power(n, 2)
                                + 4.0 * np.power(n, 3)
                                + 5.0 * np.power(n, 4)
                            )
                            * S1
                        )
                        / (n * (1.0 + n))
                    )
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                1.3333333333333333
                * (
                    -1664.0
                    - 6240.0 * n
                    - 12272.0 * np.power(n, 2)
                    - 16088.0 * np.power(n, 3)
                    - 11660.0 * np.power(n, 4)
                    - 3976.0 * np.power(n, 5)
                    + 1084.0 * np.power(n, 6)
                    + 3411.0 * np.power(n, 7)
                    + 2811.0 * np.power(n, 8)
                    + 1049.0 * np.power(n, 9)
                    + 153.0 * np.power(n, 10)
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
                    2.6666666666666665
                    * (
                        896.0
                        + 4672.0 * n
                        + 10880.0 * np.power(n, 2)
                        + 16352.0 * np.power(n, 3)
                        + 16824.0 * np.power(n, 4)
                        + 16388.0 * np.power(n, 5)
                        + 15420.0 * np.power(n, 6)
                        + 11172.0 * np.power(n, 7)
                        + 7260.0 * np.power(n, 8)
                        + 4893.0 * np.power(n, 9)
                        + 2549.0 * np.power(n, 10)
                        + 819.0 * np.power(n, 11)
                        + 151.0 * np.power(n, 12)
                        + 12.0 * np.power(n, 13)
                    )
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 5)
                    * np.power(1.0 + n, 5)
                    * np.power(2.0 + n, 4)
                )
                - (
                    1.3333333333333333
                    * (
                        -288.0
                        - 904.0 * n
                        - 844.0 * np.power(n, 2)
                        - 530.0 * np.power(n, 3)
                        - 159.0 * np.power(n, 4)
                        + 229.0 * np.power(n, 5)
                        + 271.0 * np.power(n, 6)
                        + 81.0 * np.power(n, 7)
                    )
                    * S2
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 2)
                )
            )
            + (
                10.666666666666666
                * (
                    88.0
                    + 180.0 * n
                    + 250.0 * np.power(n, 2)
                    + 283.0 * np.power(n, 3)
                    + 114.0 * np.power(n, 4)
                    + 59.0 * np.power(n, 5)
                    + 84.0 * np.power(n, 6)
                    + 40.0 * np.power(n, 7)
                    + 6.0 * np.power(n, 8)
                )
                * S21
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                1.7777777777777777
                * (
                    56.0
                    + 444.0 * n
                    - 1074.0 * np.power(n, 2)
                    - 2859.0 * np.power(n, 3)
                    - 2063.0 * np.power(n, 4)
                    - 663.0 * np.power(n, 5)
                    + 293.0 * np.power(n, 6)
                    + 478.0 * np.power(n, 7)
                    + 216.0 * np.power(n, 8)
                    + 36.0 * np.power(n, 9)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            # + (
            #     np.power(2.0, 5.0 - 1.0 * n)
            #     * (
            #         4.0
            #         - 2.0 * n
            #         + 10.0 * np.power(n, 2)
            #         - 1.0 * np.power(n, 3)
            #         + np.power(n, 5)
            #     )
            #     * (8.41439832211716 + H1)
            # )
            + np.power(2.0, 5.0)
            * H1fit
            * (4.0 - 2.0 * n + 10.0 * np.power(n, 2) - np.power(n, 3) + np.power(n, 5))
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 2))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    212.26414844076453
                    - 0.2222222222222222 * np.power(S1, 4)
                    + 1.2020569031595942 * 37.333333333333336 * S1  # - 448.0 * S1l05)
                    - 6.666666666666667 * np.power(S1, 2) * S2
                    + 15.333333333333334 * np.power(S2, 2)
                    + 1.6449340668482262 * (-4.0 * np.power(S1, 2) + 12.0 * S2)
                    + 138.66666666666666 * S211
                    + S1 * (-64.0 * S21 + 8.88888888888889 * S3)
                    # + 64.0 * (H2)
                    + 41.333333333333336 * S4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + H3fit
        )
        + 2.0
        * (
            (
                -1.0684950250307503
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 80.0 * n
                    - 200.0 * np.power(n, 2)
                    + 68.0 * np.power(n, 3)
                    + 75.0 * np.power(n, 4)
                    + 6.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                0.7310818074881006
                * (
                    864.0
                    + 4656.0 * n
                    + 9680.0 * np.power(n, 2)
                    + 12552.0 * np.power(n, 3)
                    + 9334.0 * np.power(n, 4)
                    + 4491.0 * np.power(n, 5)
                    - 934.0 * np.power(n, 6)
                    - 1109.0 * np.power(n, 7)
                    + 2196.0 * np.power(n, 8)
                    + 2251.0 * np.power(n, 9)
                    + 820.0 * np.power(n, 10)
                    + 127.0 * np.power(n, 11)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                2.6666666666666665
                * (
                    384.0
                    + 3584.0 * n
                    + 14816.0 * np.power(n, 2)
                    + 34528.0 * np.power(n, 3)
                    + 46456.0 * np.power(n, 4)
                    + 32640.0 * np.power(n, 5)
                    + 5554.0 * np.power(n, 6)
                    - 11770.0 * np.power(n, 7)
                    - 27469.0 * np.power(n, 8)
                    - 36527.0 * np.power(n, 9)
                    - 17182.0 * np.power(n, 10)
                    + 11176.0 * np.power(n, 11)
                    + 19051.0 * np.power(n, 12)
                    + 11527.0 * np.power(n, 13)
                    + 4188.0 * np.power(n, 14)
                    + 1030.0 * np.power(n, 15)
                    + 162.0 * np.power(n, 16)
                    + 12.0 * np.power(n, 17)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 5)
            )
            - (
                2.193245422464302
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 80.0 * n
                    - 20.0 * np.power(n, 2)
                    + 149.0 * np.power(n, 3)
                    + 75.0 * np.power(n, 4)
                    + 27.0 * np.power(n, 5)
                    + 17.0 * np.power(n, 6)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                8.0
                * (2.0 + n + np.power(n, 2))
                * (
                    32.0
                    + 72.0 * n
                    + 396.0 * np.power(n, 2)
                    + 810.0 * np.power(n, 3)
                    + 759.0 * np.power(n, 4)
                    + 386.0 * np.power(n, 5)
                    + 117.0 * np.power(n, 6)
                    + 22.0 * np.power(n, 7)
                    + 2.0 * np.power(n, 8)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    + 16.0 * n
                    + 18.0 * np.power(n, 2)
                    + 4.0 * np.power(n, 3)
                    + 7.0 * np.power(n, 4)
                    + 6.0 * np.power(n, 5)
                    + np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 11.0 * n + 8.0 * np.power(n, 2) + np.power(n, 3))
                * np.power(S1, 3)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                1.3333333333333333
                * (
                    384.0
                    + 2432.0 * n
                    + 6512.0 * np.power(n, 2)
                    + 9608.0 * np.power(n, 3)
                    + 8076.0 * np.power(n, 4)
                    + 3318.0 * np.power(n, 5)
                    - 2510.0 * np.power(n, 6)
                    - 3801.0 * np.power(n, 7)
                    - 1152.0 * np.power(n, 8)
                    + 104.0 * np.power(n, 9)
                    + 66.0 * np.power(n, 10)
                    + 3.0 * np.power(n, 11)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                8.0
                * (2.0 + n + np.power(n, 2))
                * (-2.0 - 27.0 * n - 12.0 * np.power(n, 2) + 3.0 * np.power(n, 3))
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 72.0 * n
                    - 56.0 * np.power(n, 2)
                    - 25.0 * np.power(n, 3)
                    - 7.0 * np.power(n, 4)
                    + 29.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    20.0
                    + 22.0 * n
                    + 7.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * (26.31894506957162 + 32.0 * Sm2)
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
            + (
                (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * (
                    -28.849365675830263
                    - 52.63789013914324 * S1
                    - 64.0 * S1 * Sm2
                    + 64.0 * Sm21
                    - 32.0 * Sm3
                )
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -6.410970150184502 * S1
                    - 0.6666666666666666 * np.power(S1, 4)
                    - 20.0 * np.power(S1, 2) * S2
                    - 2.0 * np.power(S2, 2)
                    + 16.0 * S211
                    + 16.0 * S31
                    - 36.0 * S4
                    + 1.6449340668482262
                    * (-4.0 * np.power(S1, 2) - 12.0 * S2 - 24.0 * Sm2)
                    + (-32.0 * np.power(S1, 2) - 32.0 * S2) * Sm2
                    + S1 * (-53.333333333333336 * S3 + 64.0 * Sm21)
                    - 64.0 * Sm211
                    + 32.0 * Sm22
                    - 32.0 * S1 * Sm3
                    + 32.0 * Sm31
                    - 16.0 * Sm4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 2.0
        * (
            (
                -1.0684950250307503
                * (
                    1032.0
                    + 260.0 * n
                    + 1098.0 * np.power(n, 2)
                    - 837.0 * np.power(n, 3)
                    - 5661.0 * np.power(n, 4)
                    - 472.0 * np.power(n, 5)
                    + 1135.0 * np.power(n, 6)
                    - 367.0 * np.power(n, 7)
                    - 229.0 * np.power(n, 8)
                    + 9.0 * np.power(n, 10)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.7310818074881006
                * (
                    864.0
                    + 4656.0 * n
                    + 9680.0 * np.power(n, 2)
                    + 11112.0 * np.power(n, 3)
                    + 8470.0 * np.power(n, 4)
                    + 4779.0 * np.power(n, 5)
                    - 106.0 * np.power(n, 6)
                    - 317.0 * np.power(n, 7)
                    + 2484.0 * np.power(n, 8)
                    + 2323.0 * np.power(n, 9)
                    + 856.0 * np.power(n, 10)
                    + 127.0 * np.power(n, 11)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                0.03292181069958848
                * (
                    155520.0
                    + 1.308096e6 * n
                    + 4.812768e6 * np.power(n, 2)
                    + 1.012152e7 * np.power(n, 3)
                    + 1.3312808e7 * np.power(n, 4)
                    + 1.2149124e7 * np.power(n, 5)
                    + 9.141018e6 * np.power(n, 6)
                    + 6.186057e6 * np.power(n, 7)
                    + 1.320584e6 * np.power(n, 8)
                    - 3.045065e6 * np.power(n, 9)
                    - 2.526162e6 * np.power(n, 10)
                    + 374900.0 * np.power(n, 11)
                    + 1.654143e6 * np.power(n, 12)
                    + 1.331937e6 * np.power(n, 13)
                    + 671488.0 * np.power(n, 14)
                    + 218915.0 * np.power(n, 15)
                    + 40465.0 * np.power(n, 16)
                    + 3244.0 * np.power(n, 17)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 5)
            )
            - (
                0.14814814814814814
                * (
                    1872.0
                    + 4512.0 * n
                    + 3200.0 * np.power(n, 2)
                    - 6636.0 * np.power(n, 3)
                    - 14165.0 * np.power(n, 4)
                    - 12231.0 * np.power(n, 5)
                    - 4318.0 * np.power(n, 6)
                    + 1411.0 * np.power(n, 7)
                    + 1566.0 * np.power(n, 8)
                    + 406.0 * np.power(n, 9)
                    + 145.0 * np.power(n, 10)
                    + 46.0 * np.power(n, 11)
                )
                * np.power(S1, 2)
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                0.14814814814814814
                * (
                    12240.0
                    + 55200.0 * n
                    + 106112.0 * np.power(n, 2)
                    + 114180.0 * np.power(n, 3)
                    + 81499.0 * np.power(n, 4)
                    + 62901.0 * np.power(n, 5)
                    + 17000.0 * np.power(n, 6)
                    - 773.0 * np.power(n, 7)
                    + 26208.0 * np.power(n, 8)
                    + 27688.0 * np.power(n, 9)
                    + 10993.0 * np.power(n, 10)
                    + 1696.0 * np.power(n, 11)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + S1
            * (
                (
                    0.04938271604938271
                    * (
                        1728.0
                        - 22752.0 * n
                        - 61248.0 * np.power(n, 2)
                        + 113008.0 * np.power(n, 3)
                        + 571260.0 * np.power(n, 4)
                        + 528058.0 * np.power(n, 5)
                        + 1854.0 * np.power(n, 6)
                        - 144034.0 * np.power(n, 7)
                        + 15119.0 * np.power(n, 8)
                        + 66314.0 * np.power(n, 9)
                        + 47061.0 * np.power(n, 10)
                        + 29936.0 * np.power(n, 11)
                        + 12147.0 * np.power(n, 12)
                        + 2518.0 * np.power(n, 13)
                        + 247.0 * np.power(n, 14)
                    )
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 5)
                    * np.power(1.0 + n, 5)
                    * np.power(2.0 + n, 4)
                )
                + (
                    0.4444444444444444
                    * (
                        -864.0
                        - 2560.0 * n
                        + 516.0 * np.power(n, 2)
                        + 1896.0 * np.power(n, 3)
                        + 3273.0 * np.power(n, 4)
                        + 2552.0 * np.power(n, 5)
                        + 1342.0 * np.power(n, 6)
                        + 1064.0 * np.power(n, 7)
                        + 269.0 * np.power(n, 8)
                    )
                    * S2
                )
                / (
                    np.power(-1.0 + n, 2)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 2)
                )
            )
            - (
                2.6666666666666665
                * (
                    152.0
                    + 356.0 * n
                    + 626.0 * np.power(n, 2)
                    + 763.0 * np.power(n, 3)
                    + 194.0 * np.power(n, 4)
                    - 5.0 * np.power(n, 5)
                    + 100.0 * np.power(n, 6)
                    + 48.0 * np.power(n, 7)
                    + 6.0 * np.power(n, 8)
                )
                * S21
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                0.2962962962962963
                * (
                    -2136.0
                    - 10516.0 * n
                    - 11598.0 * np.power(n, 2)
                    - 9939.0 * np.power(n, 3)
                    - 4923.0 * np.power(n, 4)
                    + 2618.0 * np.power(n, 5)
                    + 1345.0 * np.power(n, 6)
                    + 2039.0 * np.power(n, 7)
                    + 1745.0 * np.power(n, 8)
                    + 702.0 * np.power(n, 9)
                    + 135.0 * np.power(n, 10)
                )
                * S3
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            # + (
            #     np.power(2.0, 4.0 - 1.0 * n)
            #     * (
            #         4.0
            #         - 2.0 * n
            #         + 10.0 * np.power(n, 2)
            #         - 1.0 * np.power(n, 3)
            #         + np.power(n, 5)
            #     )
            #     * (-8.41439832211716 - H1)
            # )
            - np.power(2.0, 4.0)
            * H1fit
            * (4.0 - 2.0 * n + 10.0 * np.power(n, 2) - np.power(n, 3) + np.power(n, 5))
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 2))
            + (
                (
                    -2.6666666666666665
                    * (
                        -416.0
                        - 1952.0 * n
                        - 3680.0 * np.power(n, 2)
                        - 2096.0 * np.power(n, 3)
                        - 346.0 * np.power(n, 4)
                        + 14.0 * np.power(n, 5)
                        + 259.0 * np.power(n, 6)
                        + 214.0 * np.power(n, 7)
                        + 82.0 * np.power(n, 8)
                        + 44.0 * np.power(n, 9)
                        + 5.0 * np.power(n, 10)
                    )
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 4)
                    * np.power(1.0 + n, 4)
                    * np.power(2.0 + n, 3)
                )
                + (
                    10.666666666666666
                    * (
                        32.0
                        + 120.0 * n
                        + 104.0 * np.power(n, 2)
                        + 154.0 * np.power(n, 3)
                        + 122.0 * np.power(n, 4)
                        + 49.0 * np.power(n, 5)
                        + 24.0 * np.power(n, 6)
                        + 3.0 * np.power(n, 7)
                    )
                    * S1
                )
                / (
                    (-1.0 + n)
                    * np.power(n, 3)
                    * np.power(1.0 + n, 3)
                    * np.power(2.0 + n, 2)
                )
            )
            * Sm2
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -184.0593470475842
                    + 0.2222222222222222 * np.power(S1, 4)
                    # + 269.2607463077491 * S1l05
                    + 22.666666666666668 * np.power(S1, 2) * S2
                    - 26.666666666666668 * S211
                    # + 32.0 * (-H2)
                    + 37.333333333333336 * np.power(S1, 2) * Sm2
                    + 1.6449340668482262
                    * (4.0 * np.power(S1, 2) + 12.0 * S2 + 24.0 * Sm2)
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + H2fit
            + (
                5.333333333333333
                * (
                    -80.0
                    - 264.0 * n
                    - 248.0 * np.power(n, 2)
                    - 338.0 * np.power(n, 3)
                    - 293.0 * np.power(n, 4)
                    - 91.0 * np.power(n, 5)
                    + 76.0 * np.power(n, 6)
                    + 105.0 * np.power(n, 7)
                    + 39.0 * np.power(n, 8)
                    + 6.0 * np.power(n, 9)
                )
                * Sm21
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                2.6666666666666665
                * (
                    112.0
                    + 440.0 * n
                    + 248.0 * np.power(n, 2)
                    + 286.0 * np.power(n, 3)
                    + 147.0 * np.power(n, 4)
                    + 85.0 * np.power(n, 5)
                    + 148.0 * np.power(n, 6)
                    + 89.0 * np.power(n, 7)
                    + 39.0 * np.power(n, 8)
                    + 6.0 * np.power(n, 9)
                )
                * Sm3
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
                    -3.205485075092251 * (10.0 + 11.0 * n + 11.0 * np.power(n, 2)) * S1
                    + (
                        2.193245422464302
                        * (
                            -24.0
                            - 80.0 * n
                            + 76.0 * np.power(n, 2)
                            + 77.0 * np.power(n, 3)
                            + 27.0 * np.power(n, 4)
                            + 51.0 * np.power(n, 5)
                            + 17.0 * np.power(n, 6)
                        )
                        * S1
                    )
                    / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + (
                        0.14814814814814814
                        * (
                            -24.0
                            - 80.0 * n
                            + 76.0 * np.power(n, 2)
                            + 77.0 * np.power(n, 3)
                            + 27.0 * np.power(n, 4)
                            + 51.0 * np.power(n, 5)
                            + 17.0 * np.power(n, 6)
                        )
                        * np.power(S1, 3)
                    )
                    / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                    + 0.6666666666666666
                    * (74.0 + 29.0 * n + 29.0 * np.power(n, 2))
                    * np.power(S2, 2)
                    - 8.0 * (26.0 + 7.0 * n + 7.0 * np.power(n, 2)) * S31
                    + 1.3333333333333333
                    * (310.0 + 143.0 * n + 143.0 * np.power(n, 2))
                    * S4
                    + 21.333333333333332
                    * (13.0 + 7.0 * n + 7.0 * np.power(n, 2))
                    * S2
                    * Sm2
                    - 5.333333333333333
                    * (-2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                    * np.power(Sm2, 2)
                    + S1
                    * (
                        0.8888888888888888
                        * (334.0 + 137.0 * n + 137.0 * np.power(n, 2))
                        * S3
                        - 5.333333333333333
                        * (18.0 + 35.0 * n + 35.0 * np.power(n, 2))
                        * Sm21
                    )
                    + 21.333333333333332
                    * (2.0 + 13.0 * n + 13.0 * np.power(n, 2))
                    * Sm211
                    - 64.0 * (2.0 + 3.0 * n + 3.0 * np.power(n, 2)) * Sm22
                    + 2.6666666666666665
                    * (94.0 + 69.0 * n + 69.0 * np.power(n, 2))
                    * S1
                    * Sm3
                    - 10.666666666666666
                    * (22.0 + 23.0 * n + 23.0 * np.power(n, 2))
                    * Sm31
                    + 5.333333333333333
                    * (50.0 + 31.0 * n + 31.0 * np.power(n, 2))
                    * Sm4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_Hq_l3 = (
        (4.7407407407407405 * np.power(2.0 + n + np.power(n, 2), 2))
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (1.1851851851851851 * np.power(2.0 + n + np.power(n, 2), 2) * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        - 0.8888888888888888
        * (
            (
                1.3333333333333333
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 2.0
        * (
            (
                0.8888888888888888
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -12.0
                    - 34.0 * n
                    - 23.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_Hq_l2 = (
        0.3333333333333333
        * nf
        * (
            (
                3.5555555555555554
                * (
                    -24.0
                    - 20.0 * n
                    + 58.0 * np.power(n, 2)
                    + 61.0 * np.power(n, 3)
                    + 85.0 * np.power(n, 4)
                    + 83.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (10.666666666666666 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                -3.5555555555555554
                * (
                    24.0
                    + 124.0 * n
                    + 162.0 * np.power(n, 2)
                    + 193.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    + 29.0 * np.power(n, 5)
                    + 8.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (10.666666666666666 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    - 20.0 * n
                    - 26.0 * np.power(n, 2)
                    - 23.0 * np.power(n, 3)
                    + 7.0 * np.power(n, 4)
                    + 15.0 * np.power(n, 5)
                    + 7.0 * np.power(n, 6)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                8.0
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (-2.0 + n + 5.0 * np.power(n, 2))
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (16.0 * np.power(2.0 + n + np.power(n, 2), 2) * S2)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 2.0
        * (
            (
                -0.8888888888888888
                * (
                    576.0
                    + 2832.0 * n
                    + 4976.0 * np.power(n, 2)
                    + 4392.0 * np.power(n, 3)
                    + 2476.0 * np.power(n, 4)
                    + 1917.0 * np.power(n, 5)
                    + 1457.0 * np.power(n, 6)
                    + 2428.0 * np.power(n, 7)
                    + 3402.0 * np.power(n, 8)
                    + 2281.0 * np.power(n, 9)
                    + 793.0 * np.power(n, 10)
                    + 118.0 * np.power(n, 11)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 80.0 * n
                    + 40.0 * np.power(n, 2)
                    + 89.0 * np.power(n, 3)
                    + 51.0 * np.power(n, 4)
                    + 51.0 * np.power(n, 5)
                    + 17.0 * np.power(n, 6)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (np.power(2.0 + n + np.power(n, 2), 2) * (16.0 * S2 + 32.0 * Sm2))
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_Hq_l1 = (
        0.3333333333333333
        * (
            (
                2.3703703703703702
                * (
                    144.0
                    + 336.0 * n
                    + 352.0 * np.power(n, 2)
                    + 820.0 * np.power(n, 3)
                    + 2379.0 * np.power(n, 4)
                    + 2874.0 * np.power(n, 5)
                    + 2431.0 * np.power(n, 6)
                    + 1914.0 * np.power(n, 7)
                    + 1059.0 * np.power(n, 8)
                    + 320.0 * np.power(n, 9)
                    + 43.0 * np.power(n, 10)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                7.111111111111111
                * (
                    -24.0
                    - 20.0 * n
                    + 58.0 * np.power(n, 2)
                    + 61.0 * np.power(n, 3)
                    + 85.0 * np.power(n, 4)
                    + 83.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                1.0
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (-10.666666666666666 * np.power(S1, 2) - 32.0 * S2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 0.3333333333333333
        * nf
        * (
            (
                -1.1851851851851851
                * (
                    288.0
                    + 672.0 * n
                    + 16.0 * np.power(n, 2)
                    - 1232.0 * np.power(n, 3)
                    - 654.0 * np.power(n, 4)
                    - 510.0 * np.power(n, 5)
                    - 218.0 * np.power(n, 6)
                    + 912.0 * np.power(n, 7)
                    + 939.0 * np.power(n, 8)
                    + 320.0 * np.power(n, 9)
                    + 43.0 * np.power(n, 10)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                3.5555555555555554
                * (
                    -48.0
                    - 104.0 * n
                    - 56.0 * np.power(n, 2)
                    - 86.0 * np.power(n, 3)
                    - 11.0 * np.power(n, 4)
                    + 68.0 * np.power(n, 5)
                    + 37.0 * np.power(n, 6)
                    + 8.0 * np.power(n, 7)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (-5.333333333333333 * np.power(S1, 2) - 26.666666666666668 * S2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (
                -4.0
                * (
                    288.0
                    + 1712.0 * n
                    + 4656.0 * np.power(n, 2)
                    + 8248.0 * np.power(n, 3)
                    + 10938.0 * np.power(n, 4)
                    + 10519.0 * np.power(n, 5)
                    + 7642.0 * np.power(n, 6)
                    + 5020.0 * np.power(n, 7)
                    + 3520.0 * np.power(n, 8)
                    + 2328.0 * np.power(n, 9)
                    + 1107.0 * np.power(n, 10)
                    + 305.0 * np.power(n, 11)
                    + 37.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 3)
            )
            + (
                8.0
                * (
                    192.0
                    + 768.0 * n
                    + 1488.0 * np.power(n, 2)
                    + 1784.0 * np.power(n, 3)
                    + 1560.0 * np.power(n, 4)
                    + 822.0 * np.power(n, 5)
                    + 454.0 * np.power(n, 6)
                    + 567.0 * np.power(n, 7)
                    + 427.0 * np.power(n, 8)
                    + 143.0 * np.power(n, 9)
                    + 19.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (6.0 + 9.0 * n + 4.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                4.0
                * (
                    -96.0
                    - 296.0 * n
                    - 500.0 * np.power(n, 2)
                    - 658.0 * np.power(n, 3)
                    - 449.0 * np.power(n, 4)
                    - 133.0 * np.power(n, 5)
                    - 15.0 * np.power(n, 6)
                    + 3.0 * np.power(n, 7)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    115.39746270332105
                    + 2.6666666666666665 * np.power(S1, 3)
                    - 24.0 * S1 * S2
                    + 32.0 * S21
                    - 26.666666666666668 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 2.0
        * (
            (
                0.2962962962962963
                * (
                    -8640.0
                    - 56448.0 * n
                    - 150864.0 * np.power(n, 2)
                    - 225808.0 * np.power(n, 3)
                    - 250212.0 * np.power(n, 4)
                    - 241600.0 * np.power(n, 5)
                    - 206883.0 * np.power(n, 6)
                    - 156761.0 * np.power(n, 7)
                    - 125240.0 * np.power(n, 8)
                    - 72944.0 * np.power(n, 9)
                    + 9045.0 * np.power(n, 10)
                    + 43489.0 * np.power(n, 11)
                    + 25572.0 * np.power(n, 12)
                    + 6560.0 * np.power(n, 13)
                    + 686.0 * np.power(n, 14)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            - (
                0.8888888888888888
                * (
                    72.0
                    + 924.0 * n
                    + 418.0 * np.power(n, 2)
                    - 3167.0 * np.power(n, 3)
                    - 3105.0 * np.power(n, 4)
                    - 2106.0 * np.power(n, 5)
                    - 2555.0 * np.power(n, 6)
                    - 438.0 * np.power(n, 7)
                    + 1110.0 * np.power(n, 8)
                    + 647.0 * np.power(n, 9)
                    + 136.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 2)
            )
            + (
                1.3333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 16.0 * n
                    + 41.0 * np.power(n, 2)
                    - 6.0 * np.power(n, 3)
                    + 17.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 2)
                * (2.0 + n)
            )
            + (
                1.3333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    -120.0
                    - 412.0 * n
                    - 238.0 * np.power(n, 2)
                    + 31.0 * np.power(n, 3)
                    + 45.0 * np.power(n, 4)
                    + 189.0 * np.power(n, 5)
                    + 73.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (74.0 + 31.0 * n + 31.0 * np.power(n, 2))
                * S3
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                32.0
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * Sm2
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            + (
                16.0
                * (
                    40.0
                    + 132.0 * n
                    + 158.0 * np.power(n, 2)
                    + 155.0 * np.power(n, 3)
                    + 102.0 * np.power(n, 4)
                    + 37.0 * np.power(n, 5)
                    + 14.0 * np.power(n, 6)
                    + 2.0 * np.power(n, 7)
                )
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (128.0 * (1.0 + n + np.power(n, 2)) * (2.0 + n + np.power(n, 2)) * Sm21)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                16.0
                * (2.0 + n + np.power(n, 2))
                * (10.0 + 7.0 * n + 7.0 * np.power(n, 2))
                * Sm3
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -115.39746270332105
                    - 2.6666666666666665 * np.power(S1, 3)
                    + 40.0 * S1 * S2
                    - 32.0 * S21
                    + 64.0 * S1 * Sm2
                    + 16.0 * Sm3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    return a_Hq_l0 + a_Hq_l1 * L + a_Hq_l2 * L**2 + a_Hq_l3 * L**3
