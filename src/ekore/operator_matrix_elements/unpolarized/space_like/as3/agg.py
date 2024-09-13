"""The unpolarized, space-like |N3LO| gluon-gluon |OME|."""

# pylint: skip-file
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .....harmonics.log_functions import (
    lm11,
    lm11m1,
    lm11m2,
    lm12,
    lm12m1,
    lm12m2,
    lm13,
    lm13m1,
    lm13m2,
    lm14m1,
    lm14m2,
)


@nb.njit(cache=True)
def a_gg3(n, cache, nf):
    r"""Compute :math:`a_{gg}^{(3)}(N)`.

    The expression is presented in  :cite:`Ablinger:2022wbb`.

    The :math:`n_f^0` piece is parametrized from:

    - the small-x limit :eqref:`4.10`
    - the large-x limit :eqref:`4.11`
    - the expansion of the local and singular parts in :eqref:`4.6, 4.7`
    - the first 15 Mellin moments up to :math:`N=30`

    The analytical expression contains binomial factors
    which are not practical to use.

    When using the code, please cite the complete list of references
    available in :mod:`~ekore.operator_matrix_elements.unpolarized.space_like.as3`.

    Parameters
    ----------
    n : complex
        Mellin moment
    cache : numpy.ndarray
        Harmonic sum cache
    nf : int
        number of active flavor below the threshold

    Returns
    -------
    complex
        :math:`a_{gg}^{(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)
    S4 = c.get(c.S4, cache, n)

    Lm11 = lm11(n, S1)
    Lm12 = lm12(n, S1, S2)
    Lm13 = lm13(n, S1, S2, S3)
    Lm11m1 = lm11m1(n, S1)
    Lm12m1 = lm12m1(n, S1, S2)
    Lm13m1 = lm13m1(n, S1, S2, S3)
    Lm14m1 = lm14m1(n, S1, S2, S3, S4)
    Lm11m2 = lm11m2(n, S1)
    Lm12m2 = lm12m2(n, S1, S2)
    Lm13m2 = lm13m2(n, S1, S2, S3)
    Lm14m2 = lm14m2(n, S1, S2, S3, S4)
    # the nf^0 part is parametrized since it contains nasty binomial factors.
    agg3_nf0_param = (
        619.2420126046355
        + 701.1986854426286 / (-1.0 + n) ** 2
        - 4954.442786280953 / (-1.0 + n)
        + 305.77777777777777 / n**6
        - 668.4444444444445 / n**5
        + 2426.352476661977 / n**4
        - 3148.735962235475 / n**3
        + 9155.33153602228 / n**2
        + 5069.820034891387 / n
        - 6471.478696979203 / (1.0 + n) ** 2
        - 8987.70366338934 / (n + n**2)
        - 21902.776840085757 / (2.0 + 3.0 * n + n**2)
        - 78877.91436146703 / (3.0 + 4.0 * n + n**2)
        - 207627.85210030797 / (6.0 + 5.0 * n + n**2)
        + 860105.1673083167 / (6.0 + 11.0 * n + 6.0 * n**2 + n**3)
        + 714.9711186248866 * S1
        + 576.0307099030653 * Lm11
        - 14825.806057017968 * Lm11m1
        + 368095.9894734118 * Lm11m2
        + 40.908173376688424 * Lm12
        - 6838.198890554838 * Lm12m1
        + 474165.7099083288 * Lm12m2
        + 5.333333333333333 * Lm13
        - 4424.7425689765805 * Lm13m1
        + 50838.65442166183 * Lm13m2
        - 508.9445773396529 * Lm14m1
        + 28154.716500168193 * Lm14m2
    )
    agg3_nf1 = 0.75 * (
        -(
            (
                0.0027434842249657062
                * (
                    -12096.0
                    - 25344 * n
                    - 576 * n**2
                    - 63040 * n**3
                    - 388726 * n**4
                    - 770095 * n**5
                    - 794647 * n**6
                    - 417598 * n**7
                    - 52924 * n**8
                    + 36045 * n**9
                    + 7209 * n**10
                )
            )
            / ((-1 + n) * n**4 * (1.0 + n) ** 4 * (2.0 + n))
        )
        + 1.6449340668482262
        * (
            (
                0.14814814814814814
                * (
                    48.0
                    + 224 * n
                    + 358 * n**2
                    + 277 * n**3
                    + 161 * n**4
                    + 27 * n**5
                    + 9 * n**6
                )
            )
            / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
            - 5.925925925925926 * S1
        )
        - (
            0.010973936899862825
            * (
                864.0
                - 2016 * n
                - 11178 * n**2
                - 27001 * n**3
                - 39319 * n**4
                - 15103 * n**5
                + 23321 * n**6
                + 26480 * n**7
                + 6944 * n**8
            )
            * S1
        )
        / ((-1.0 + n) * n**3 * (1.0 + n) ** 3 * (2.0 + n))
        - (
            0.14814814814814814
            * (32.0 + 70 * n + 47 * n**2 + 2 * n**3 + 41 * n**4 + 16 * n**5)
            * S1**2
        )
        / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
        + 1.2020569031595942
        * (
            -(
                (33.18518518518518 * (1.0 + n + n**2))
                / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            )
            + 16.59259259259259 * S1
        )
        + (
            0.14814814814814814
            * (
                -96.0
                - 210 * n
                - 301 * n**2
                - 166 * n**3
                - 3 * n**4
                + 112 * n**5
                + 40 * n**6
            )
            * S2
        )
        / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
    ) + 0.3333333333333333 * (
        -(
            (59.835721401722026 * (2.0 + n + n**2) ** 2)
            / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
        )
        - (
            0.00823045267489712
            * (
                12096.0
                + 45504 * n
                + 67728 * n**2
                - 110240 * n**3
                - 563504 * n**4
                - 867778 * n**5
                - 829641 * n**6
                - 664606 * n**7
                - 399973 * n**8
                - 121030 * n**9
                + 6253 * n**10
                + 17478 * n**11
                + 2913 * n**12
            )
        )
        / ((-1.0 + n) * n**5 * (1.0 + n) ** 5 * (2.0 + n))
        + (
            0.5925925925925926
            * (
                24.0
                + 248 * n
                + 520 * n**2
                + 543 * n**3
                + 386 * n**4
                + 123 * n**5
                + 44 * n**6
            )
            * S1**2
        )
        / ((-1.0 + n) * n**3 * (1.0 + n) ** 3 * (2.0 + n))
        - (4.148148148148148 * (2.0 + n + n**2) ** 2 * S1**3)
        / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
        + 1.6449340668482262
        * (
            -(
                (
                    0.4444444444444444
                    * (
                        48.0
                        - 80 * n
                        - 220 * n**2
                        - 186 * n**3
                        - 311 * n**4
                        - 162 * n**5
                        + 4 * n**6
                        + 60 * n**7
                        + 15 * n**8
                    )
                )
                / ((-1.0 + n) * n**3 * (1.0 + n) ** 3 * (2.0 + n))
            )
            - (5.333333333333333 * (2.0 + n + n**2) ** 2 * S1)
            / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
        )
        - (
            1.7777777777777777
            * (
                -24.0
                - 56 * n
                - 100 * n**2
                - 129 * n**3
                - 50 * n**4
                + 3 * n**5
                + 4 * n**6
            )
            * S2
        )
        / ((-1.0 + n) * n**3 * (1.0 + n) ** 3 * (2.0 + n))
        + S1
        * (
            -(
                (
                    0.3950617283950617
                    * (
                        -72.0
                        + 48 * n
                        + 1534 * n**2
                        + 4722 * n**3
                        + 7310 * n**4
                        + 6484 * n**5
                        + 3169 * n**6
                        + 856 * n**7
                        + 205 * n**8
                    )
                )
                / ((-1.0 + n) * n**4 * (1.0 + n) ** 4 * (2.0 + n))
            )
            - (5.333333333333333 * (2.0 + n + n**2) ** 2 * S2)
            / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
        )
        + (5.925925925925926 * (2.0 + n + n**2) ** 2 * S3)
        / ((-1.0 + n) * n**2 * (1.0 + n) ** 2 * (2.0 + n))
    )
    return agg3_nf0_param + agg3_nf1 * nf


@nb.njit(cache=True)
def A_gg(n, cache, nf, L):
    r"""Compute the |N3LO| singlet |OME| :math:`A_{gg}^{S,(3)}(N)`.

    The expression is presented in :cite:`Bierenbaum:2009mv`.

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
        :math:`A_{gg}^{S,(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    Sm2 = c.get(c.Sm2, cache, n, is_singlet=True)
    S3 = c.get(c.S3, cache, n)
    S21 = c.get(c.S21, cache, n)
    Sm21 = c.get(c.Sm21, cache, n, is_singlet=True)
    Sm3 = c.get(c.Sm3, cache, n, is_singlet=True)
    S4 = c.get(c.S4, cache, n)
    S31 = c.get(c.S31, cache, n)
    S211 = c.get(c.S211, cache, n)
    Sm22 = c.get(c.Sm22, cache, n, is_singlet=True)
    Sm211 = c.get(c.Sm211, cache, n, is_singlet=True)
    Sm31 = c.get(c.Sm31, cache, n, is_singlet=True)
    Sm4 = c.get(c.Sm4, cache, n, is_singlet=True)
    a_gg_l0 = (
        -0.35616500834358344
        + a_gg3(n, cache, nf)
        + 0.75
        * (
            (-19.945240467240673 * (1.0 + n + np.power(n, 2)))
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            - (
                0.24369393582936685
                * (
                    168.0
                    + 784.0 * n
                    + 1118.0 * np.power(n, 2)
                    + 767.0 * np.power(n, 3)
                    + 631.0 * np.power(n, 4)
                    + 297.0 * np.power(n, 5)
                    + 99.0 * np.power(n, 6)
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                0.09876543209876543
                * (
                    216.0
                    + 288.0 * n
                    - 30.0 * np.power(n, 2)
                    + 3556.0 * np.power(n, 3)
                    + 14212.0 * np.power(n, 4)
                    + 23815.0 * np.power(n, 5)
                    + 22951.0 * np.power(n, 6)
                    + 12778.0 * np.power(n, 7)
                    + 3316.0 * np.power(n, 8)
                    + 15.0 * np.power(n, 9)
                    + 3.0 * np.power(n, 10)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + 44.0897712497317 * S1
            + (
                0.19753086419753085
                * (
                    54.0
                    - 175.0 * n
                    - 247.0 * np.power(n, 2)
                    + 256.0 * np.power(n, 3)
                    + 328.0 * np.power(n, 4)
                )
                * S1
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            + (2.6666666666666665 * np.power(S1, 2)) / (1.0 + n)
            - (2.6666666666666665 * (1.0 + 2.0 * n) * S2) / (1.0 + n)
            + nf
            * (
                (-5.698640133497335 * (1.0 + n + np.power(n, 2)))
                / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                - (
                    0.24369393582936685
                    * (
                        48.0
                        + 224.0 * n
                        + 358.0 * np.power(n, 2)
                        + 277.0 * np.power(n, 3)
                        + 161.0 * np.power(n, 4)
                        + 27.0 * np.power(n, 5)
                        + 9.0 * np.power(n, 6)
                    )
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
                + (
                    0.13168724279835392
                    * (
                        -108.0
                        - 144.0 * n
                        + 15.0 * np.power(n, 2)
                        - 1778.0 * np.power(n, 3)
                        - 7235.0 * np.power(n, 4)
                        - 12359.0 * np.power(n, 5)
                        - 11927.0 * np.power(n, 6)
                        - 6260.0 * np.power(n, 7)
                        - 1142.0 * np.power(n, 8)
                        + 315.0 * np.power(n, 9)
                        + 63.0 * np.power(n, 10)
                    )
                )
                / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
                + 12.597077499923342 * S1
                + (
                    0.13168724279835392
                    * (
                        54.0
                        - 175.0 * n
                        - 247.0 * np.power(n, 2)
                        + 256.0 * np.power(n, 3)
                        + 328.0 * np.power(n, 4)
                    )
                    * S1
                )
                / ((-1.0 + n) * n * np.power(1.0 + n, 2))
                + (1.7777777777777777 * np.power(S1, 2)) / (1.0 + n)
                - (1.7777777777777777 * (1.0 + 2.0 * n) * S2) / (1.0 + n)
            )
        )
        + 0.3333333333333333
        * (
            (
                -1.4621636149762012
                * (
                    -84.0
                    - 148.0 * n
                    - 245.0 * np.power(n, 2)
                    - 378.0 * np.power(n, 3)
                    - 166.0 * np.power(n, 4)
                    + 12.0 * np.power(n, 5)
                    + 86.0 * np.power(n, 6)
                    + 60.0 * np.power(n, 7)
                    + 15.0 * np.power(n, 8)
                )
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                0.2222222222222222
                * (
                    288.0
                    + 864.0 * n
                    + 1224.0 * np.power(n, 2)
                    - 792.0 * np.power(n, 4)
                    + 4546.0 * np.power(n, 5)
                    + 7713.0 * np.power(n, 6)
                    + 1150.0 * np.power(n, 7)
                    - 2243.0 * np.power(n, 8)
                    + 2758.0 * np.power(n, 9)
                    + 4795.0 * np.power(n, 10)
                    + 2346.0 * np.power(n, 11)
                    + 391.0 * np.power(n, 12)
                )
            )
            / ((-1.0 + n) * np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (-10.684950250307505 - 8.772981689857207 * S1)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + nf
            * (
                (
                    0.7310818074881006
                    * (
                        48.0
                        - 80.0 * n
                        - 220.0 * np.power(n, 2)
                        - 282.0 * np.power(n, 3)
                        - 551.0 * np.power(n, 4)
                        - 258.0 * np.power(n, 5)
                        + 196.0 * np.power(n, 6)
                        + 252.0 * np.power(n, 7)
                        + 63.0 * np.power(n, 8)
                    )
                )
                / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
                + (
                    0.024691358024691357
                    * (
                        1728.0
                        + 5184.0 * n
                        + 7344.0 * np.power(n, 2)
                        - 15872.0 * np.power(n, 3)
                        - 70928.0 * np.power(n, 4)
                        - 90898.0 * np.power(n, 5)
                        - 79041.0 * np.power(n, 6)
                        - 82318.0 * np.power(n, 7)
                        - 62269.0 * np.power(n, 8)
                        - 8758.0 * np.power(n, 9)
                        + 15013.0 * np.power(n, 10)
                        + 9558.0 * np.power(n, 11)
                        + 1593.0 * np.power(n, 12)
                    )
                )
                / ((-1.0 + n) * np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
                + (
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
                - (
                    1.7777777777777777
                    * (2.0 + n + np.power(n, 2))
                    * (16.0 + 27.0 * n + 13.0 * np.power(n, 2) + 8.0 * np.power(n, 3))
                    * (np.power(S1, 2) + S2)
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
                + (
                    np.power(2.0 + n + np.power(n, 2), 2)
                    * (
                        -8.547960200246003
                        + 8.772981689857207 * S1
                        + 1.7777777777777777 * np.power(S1, 3)
                        + 5.333333333333333 * S1 * S2
                        + 3.5555555555555554 * S3
                    )
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            )
        )
        + 0.8888888888888888
        * (
            145.94322056228512
            - (
                1.6027425375461255
                * (
                    -8.0
                    - 20.0 * n
                    - 34.0 * np.power(n, 2)
                    - 79.0 * np.power(n, 3)
                    - 143.0 * np.power(n, 4)
                    - 57.0 * np.power(n, 5)
                    + 93.0 * np.power(n, 6)
                    + 96.0 * np.power(n, 7)
                    + 24.0 * np.power(n, 8)
                )
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                3.2898681336964524
                * (
                    24.0
                    + 68.0 * n
                    - 2.0 * np.power(n, 2)
                    - 205.0 * np.power(n, 3)
                    - 509.0 * np.power(n, 4)
                    - 753.0 * np.power(n, 5)
                    - 615.0 * np.power(n, 6)
                    - 66.0 * np.power(n, 7)
                    + 282.0 * np.power(n, 8)
                    + 200.0 * np.power(n, 9)
                    + 40.0 * np.power(n, 10)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                1.0
                * (
                    32.0
                    + 208.0 * n
                    + 280.0 * np.power(n, 2)
                    - 532.0 * np.power(n, 3)
                    - 1944.0 * np.power(n, 4)
                    - 1520.0 * np.power(n, 5)
                    + 2258.0 * np.power(n, 6)
                    + 6555.0 * np.power(n, 7)
                    + 6707.0 * np.power(n, 8)
                    + 3479.0 * np.power(n, 9)
                    + 1343.0 * np.power(n, 10)
                    + 1025.0 * np.power(n, 11)
                    + 741.0 * np.power(n, 12)
                    + 273.0 * np.power(n, 13)
                    + 39.0 * np.power(n, 14)
                )
            )
            / ((-1.0 + n) * np.power(n, 6) * np.power(1.0 + n, 6) * (2.0 + n))
            - (
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
            - (
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
            + (
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
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    - 20.0 * n
                    - 56.0 * np.power(n, 2)
                    - 64.0 * np.power(n, 3)
                    + 15.0 * np.power(n, 4)
                    + 30.0 * np.power(n, 5)
                    + 3.0 * np.power(n, 6)
                )
                * S2
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                (2.0 + 3.0 * n)
                * (2.0 + n + np.power(n, 2))
                * (2.6666666666666665 * np.power(S1, 3) + 8.0 * S1 * S2)
            )
            / ((-1.0 + n) * np.power(n, 3) * (1.0 + n) * (2.0 + n))
            - (
                32.0
                * (-2.0 - 3.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * S21
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
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
                    -6.410970150184502 * S1
                    - 0.6666666666666666 * np.power(S1, 4)
                    - 4.0 * np.power(S1, 2) * S2
                    - 2.0 * np.power(S2, 2)
                    + 1.6449340668482262 * (-4.0 * np.power(S1, 2) + 12.0 * S2)
                    + 64.0 * S211
                    + S1 * (-32.0 * S21 - 5.333333333333333 * S3)
                    - 32.0 * S31
                    + 12.0 * S4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 4.5
        * (
            (
                -0.03292181069958848
                * (
                    -1188.0
                    - 1584.0 * n
                    + 165.0 * np.power(n, 2)
                    - 18010.0 * np.power(n, 3)
                    - 73393.0 * np.power(n, 4)
                    - 125113.0 * np.power(n, 5)
                    - 120361.0 * np.power(n, 6)
                    - 62668.0 * np.power(n, 7)
                    - 11014.0 * np.power(n, 8)
                    + 3465.0 * np.power(n, 9)
                    + 693.0 * np.power(n, 10)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                0.24369393582936685
                * (
                    -576.0
                    - 2544.0 * n
                    - 5824.0 * np.power(n, 2)
                    - 8400.0 * np.power(n, 3)
                    - 4984.0 * np.power(n, 4)
                    + 211.0 * np.power(n, 5)
                    + 2207.0 * np.power(n, 6)
                    + 2155.0 * np.power(n, 7)
                    + 1356.0 * np.power(n, 8)
                    + 631.0 * np.power(n, 9)
                    + 189.0 * np.power(n, 10)
                    + 27.0 * np.power(n, 11)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - 7.835630183558836 * S1
            - (
                0.03292181069958848
                * (
                    594.0
                    - 1151.0 * n
                    - 1943.0 * np.power(n, 2)
                    + 2042.0 * np.power(n, 3)
                    + 2834.0 * np.power(n, 4)
                )
                * S1
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2))
            + (
                0.9747757433174674
                * (
                    -72.0
                    - 72.0 * n
                    + 256.0 * np.power(n, 2)
                    + 292.0 * np.power(n, 3)
                    + 173.0 * np.power(n, 4)
                    + 64.0 * np.power(n, 5)
                    + 2.0 * np.power(n, 6)
                    + 4.0 * np.power(n, 7)
                    + np.power(n, 8)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (4.888888888888889 * np.power(S1, 2)) / (1.0 + n)
            + (4.888888888888889 * (1.0 + 2.0 * n) * S2) / (1.0 + n)
            + (
                (1.0 + n + np.power(n, 2))
                * (
                    15.671260367117672
                    + 1.6449340668482262
                    * (21.333333333333332 * S2 + 21.333333333333332 * Sm2)
                )
            )
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            + 1.6449340668482262
            * (
                -10.666666666666666 * S1 * S2
                - 5.333333333333333 * S3
                - 10.666666666666666 * S1 * Sm2
                + 10.666666666666666 * Sm21
                - 5.333333333333333 * Sm3
            )
        )
        + 2.0
        * (
            -72.97161028114256
            + (
                1.0684950250307503
                * (
                    -48.0
                    - 184.0 * n
                    - 504.0 * np.power(n, 2)
                    - 72.0 * np.power(n, 3)
                    + 162.0 * np.power(n, 4)
                    + 101.0 * np.power(n, 5)
                    - 167.0 * np.power(n, 6)
                    - 91.0 * np.power(n, 7)
                    + 119.0 * np.power(n, 8)
                    + 90.0 * np.power(n, 9)
                    + 18.0 * np.power(n, 10)
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
                    -864.0
                    - 2544.0 * n
                    - 3344.0 * np.power(n, 2)
                    - 5880.0 * np.power(n, 3)
                    - 5230.0 * np.power(n, 4)
                    - 7911.0 * np.power(n, 5)
                    - 12524.0 * np.power(n, 6)
                    - 9220.0 * np.power(n, 7)
                    - 1680.0 * np.power(n, 8)
                    + 2042.0 * np.power(n, 9)
                    + 1562.0 * np.power(n, 10)
                    + 530.0 * np.power(n, 11)
                    + 120.0 * np.power(n, 12)
                    + 15.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                0.05555555555555555
                * (
                    18432.0
                    + 104448.0 * n
                    + 305664.0 * np.power(n, 2)
                    + 574464.0 * np.power(n, 3)
                    + 807552.0 * np.power(n, 4)
                    + 1.160704e6 * np.power(n, 5)
                    + 952768.0 * np.power(n, 6)
                    - 227344.0 * np.power(n, 7)
                    - 85568.0 * np.power(n, 8)
                    + 2.284064e6 * np.power(n, 9)
                    + 2.719198e6 * np.power(n, 10)
                    - 792201.0 * np.power(n, 11)
                    - 3.594388e6 * np.power(n, 12)
                    - 2.371724e6 * np.power(n, 13)
                    + 244448.0 * np.power(n, 14)
                    + 1.224418e6 * np.power(n, 15)
                    + 794084.0 * np.power(n, 16)
                    + 257636.0 * np.power(n, 17)
                    + 43890.0 * np.power(n, 18)
                    + 3135.0 * np.power(n, 19)
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
                * (
                    48.0
                    + 184.0 * n
                    + 144.0 * np.power(n, 2)
                    - 78.0 * np.power(n, 3)
                    - 39.0 * np.power(n, 4)
                    - 368.0 * np.power(n, 5)
                    - 586.0 * np.power(n, 6)
                    - 224.0 * np.power(n, 7)
                    + 163.0 * np.power(n, 8)
                    + 150.0 * np.power(n, 9)
                    + 30.0 * np.power(n, 10)
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
                0.2222222222222222
                * (
                    -2304.0
                    - 6336.0 * n
                    - 34832.0 * np.power(n, 2)
                    - 91776.0 * np.power(n, 3)
                    - 141176.0 * np.power(n, 4)
                    - 137724.0 * np.power(n, 5)
                    - 69461.0 * np.power(n, 6)
                    + 12096.0 * np.power(n, 7)
                    + 46703.0 * np.power(n, 8)
                    + 34680.0 * np.power(n, 9)
                    + 13349.0 * np.power(n, 10)
                    + 2724.0 * np.power(n, 11)
                    + 233.0 * np.power(n, 12)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            - (
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
            - (
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
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    64.0
                    + 256.0 * n
                    + 456.0 * np.power(n, 2)
                    + 600.0 * np.power(n, 3)
                    + 290.0 * np.power(n, 4)
                    + 42.0 * np.power(n, 5)
                    + 105.0 * np.power(n, 6)
                    + 85.0 * np.power(n, 7)
                    + 21.0 * np.power(n, 8)
                    + np.power(n, 9)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
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
                21.333333333333332
                * (2.0 + n + np.power(n, 2))
                * (
                    6.0
                    + 7.0 * n
                    + 3.0 * np.power(n, 2)
                    + 9.0 * np.power(n, 3)
                    + 10.0 * np.power(n, 4)
                    + np.power(n, 5)
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
                * (-26.31894506957162 - 32.0 * Sm2)
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
            + (
                (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * (
                    28.849365675830263
                    + 52.63789013914324 * S1
                    + 64.0 * S1 * Sm2
                    - 64.0 * Sm21
                    + 32.0 * Sm3
                )
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    6.410970150184502 * S1
                    + 0.6666666666666666 * np.power(S1, 4)
                    + 20.0 * np.power(S1, 2) * S2
                    + 2.0 * np.power(S2, 2)
                    - 16.0 * S211
                    - 16.0 * S31
                    + 36.0 * S4
                    + (32.0 * np.power(S1, 2) + 32.0 * S2) * Sm2
                    + 1.6449340668482262
                    * (4.0 * np.power(S1, 2) + 12.0 * S2 + 24.0 * Sm2)
                    + S1 * (53.333333333333336 * S3 - 64.0 * Sm21)
                    + 64.0 * Sm211
                    - 32.0 * Sm22
                    + 32.0 * S1 * Sm3
                    - 32.0 * Sm31
                    + 16.0 * Sm4
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_gg_l3 = (
        -0.2962962962962963
        + 0.3333333333333333
        * (
            (-8.88888888888889 * np.power(2.0 + n + np.power(n, 2), 2))
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (7.111111111111111 * np.power(2.0 + n + np.power(n, 2), 2) * nf)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 4.5
        * (
            (-13.037037037037036 * (1.0 + n + np.power(n, 2)))
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            + 6.518518518518518 * S1
        )
        + 0.75
        * (
            (-16.59259259259259 * (1.0 + n + np.power(n, 2)))
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            - 1.0
            * nf
            * (
                (4.7407407407407405 * (1.0 + n + np.power(n, 2)))
                / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
                - 2.3703703703703702 * S1
            )
            + 8.296296296296296 * S1
        )
        - 2.0
        * (
            (
                -0.8888888888888888
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
            - (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (
                -1.3333333333333333
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_gg_l2 = (
        0.75
        * (
            (
                0.2962962962962963
                * (
                    96.0
                    + 448.0 * n
                    + 626.0 * np.power(n, 2)
                    + 419.0 * np.power(n, 3)
                    + 367.0 * np.power(n, 4)
                    + 189.0 * np.power(n, 5)
                    + 63.0 * np.power(n, 6)
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - 23.703703703703702 * S1
        )
        + 0.3333333333333333
        * (
            (
                0.8888888888888888
                * (
                    -96.0
                    - 224.0 * n
                    - 400.0 * np.power(n, 2)
                    - 546.0 * np.power(n, 3)
                    - 275.0 * np.power(n, 4)
                    - 18.0 * np.power(n, 5)
                    + 76.0 * np.power(n, 6)
                    + 60.0 * np.power(n, 7)
                    + 15.0 * np.power(n, 8)
                )
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (10.666666666666666 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                -4.0
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
            + (
                8.0
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (-2.0 + n + 5.0 * np.power(n, 2))
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (16.0 * np.power(2.0 + n + np.power(n, 2), 2) * S2)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 2.0
        * (
            (
                -0.2222222222222222
                * (
                    -2304.0
                    - 7104.0 * n
                    - 7232.0 * np.power(n, 2)
                    - 2496.0 * np.power(n, 3)
                    + 5240.0 * np.power(n, 4)
                    - 3624.0 * np.power(n, 5)
                    - 26198.0 * np.power(n, 6)
                    - 34351.0 * np.power(n, 7)
                    - 23124.0 * np.power(n, 8)
                    - 8809.0 * np.power(n, 9)
                    - 1366.0 * np.power(n, 10)
                    + 479.0 * np.power(n, 11)
                    + 264.0 * np.power(n, 12)
                    + 33.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                2.6666666666666665
                * (
                    -48.0
                    - 184.0 * n
                    - 24.0 * np.power(n, 2)
                    + 150.0 * np.power(n, 3)
                    + 255.0 * np.power(n, 4)
                    + 233.0 * np.power(n, 5)
                    + 91.0 * np.power(n, 6)
                    + 50.0 * np.power(n, 7)
                    + 35.0 * np.power(n, 8)
                    + 15.0 * np.power(n, 9)
                    + 3.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (np.power(2.0 + n + np.power(n, 2), 2) * (-16.0 * S2 - 32.0 * Sm2))
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        + 4.5
        * (
            (
                -0.2222222222222222
                * (
                    -768.0
                    - 2688.0 * n
                    - 4256.0 * np.power(n, 2)
                    - 4632.0 * np.power(n, 3)
                    - 2060.0 * np.power(n, 4)
                    - 934.0 * np.power(n, 5)
                    - 2099.0 * np.power(n, 6)
                    - 2185.0 * np.power(n, 7)
                    - 1014.0 * np.power(n, 8)
                    - 124.0 * np.power(n, 9)
                    + 21.0 * np.power(n, 10)
                    + 3.0 * np.power(n, 11)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.8888888888888888
                * (
                    -96.0
                    - 96.0 * n
                    + 428.0 * np.power(n, 2)
                    + 476.0 * np.power(n, 3)
                    + 79.0 * np.power(n, 4)
                    - 88.0 * np.power(n, 5)
                    + 46.0 * np.power(n, 6)
                    + 92.0 * np.power(n, 7)
                    + 23.0 * np.power(n, 8)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + 21.333333333333332 * S1 * S2
            + 10.666666666666666 * S3
            + (
                (1.0 + n + np.power(n, 2))
                * (-42.666666666666664 * S2 - 42.666666666666664 * Sm2)
            )
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            + 21.333333333333332 * S1 * Sm2
            - 21.333333333333332 * Sm21
            + 10.666666666666666 * Sm3
        )
    )
    a_gg_l1 = (
        0.75
        * (
            (
                0.07407407407407407
                * (
                    480.0
                    + 1184.0 * n
                    - 256.0 * np.power(n, 2)
                    - 3762.0 * np.power(n, 3)
                    - 5931.0 * np.power(n, 4)
                    - 4554.0 * np.power(n, 5)
                    - 1440.0 * np.power(n, 6)
                    + 108.0 * np.power(n, 7)
                    + 27.0 * np.power(n, 8)
                )
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                0.8888888888888888
                * (
                    -32.0
                    - 70.0 * n
                    - 147.0 * np.power(n, 2)
                    - 132.0 * np.power(n, 3)
                    + 19.0 * np.power(n, 4)
                    + 114.0 * np.power(n, 5)
                    + 40.0 * np.power(n, 6)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - 1.0
            * nf
            * (
                (
                    -0.04938271604938271
                    * (
                        864.0
                        + 3360.0 * n
                        + 5008.0 * np.power(n, 2)
                        + 2874.0 * np.power(n, 3)
                        - 1193.0 * np.power(n, 4)
                        - 2094.0 * np.power(n, 5)
                        + 640.0 * np.power(n, 6)
                        + 1188.0 * np.power(n, 7)
                        + 297.0 * np.power(n, 8)
                    )
                )
                / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
                - (
                    0.19753086419753085
                    * (
                        -288.0
                        - 630.0 * n
                        - 947.0 * np.power(n, 2)
                        - 552.0 * np.power(n, 3)
                        + 19.0 * np.power(n, 4)
                        + 390.0 * np.power(n, 5)
                        + 136.0 * np.power(n, 6)
                    )
                    * S1
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            )
        )
        + 0.3333333333333333
        * (
            (
                0.2962962962962963
                * (
                    -360.0
                    - 1488.0 * n
                    - 3466.0 * np.power(n, 2)
                    - 4326.0 * np.power(n, 3)
                    - 3242.0 * np.power(n, 4)
                    - 2947.0 * np.power(n, 5)
                    - 2467.0 * np.power(n, 6)
                    - 82.0 * np.power(n, 7)
                    + 1640.0 * np.power(n, 8)
                    + 1095.0 * np.power(n, 9)
                    + 219.0 * np.power(n, 10)
                )
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                3.5555555555555554
                * (
                    -24.0
                    - 56.0 * n
                    - 100.0 * np.power(n, 2)
                    - 129.0 * np.power(n, 3)
                    - 50.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                    + 4.0 * np.power(n, 6)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - 1.0
            * nf
            * (
                (
                    -0.4444444444444444
                    * (
                        -288.0
                        - 1600.0 * n
                        - 4648.0 * np.power(n, 2)
                        - 7656.0 * np.power(n, 3)
                        - 8310.0 * np.power(n, 4)
                        - 6669.0 * np.power(n, 5)
                        - 3349.0 * np.power(n, 6)
                        - 762.0 * np.power(n, 7)
                        + 368.0 * np.power(n, 8)
                        + 335.0 * np.power(n, 9)
                        + 67.0 * np.power(n, 10)
                    )
                )
                / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
                - (
                    10.666666666666666
                    * (
                        16.0
                        + 48.0 * n
                        + 90.0 * np.power(n, 2)
                        + 109.0 * np.power(n, 3)
                        + 52.0 * np.power(n, 4)
                        + 5.0 * np.power(n, 5)
                    )
                    * S1
                )
                / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
                + (
                    np.power(2.0 + n + np.power(n, 2), 2)
                    * (16.0 * np.power(S1, 2) - 26.666666666666668 * S2)
                )
                / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            )
            - (
                1.0
                * np.power(2.0 + n + np.power(n, 2), 2)
                * (5.333333333333333 * np.power(S1, 2) - 16.0 * S2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (-115.39746270332105 * (2.0 + n + np.power(n, 2)))
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                2.0
                * (
                    48.0
                    + 152.0 * n
                    + 324.0 * np.power(n, 2)
                    + 490.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    - 786.0 * np.power(n, 5)
                    - 1057.0 * np.power(n, 6)
                    - 846.0 * np.power(n, 7)
                    - 379.0 * np.power(n, 8)
                    - 80.0 * np.power(n, 9)
                    - 5.0 * np.power(n, 10)
                    + 6.0 * np.power(n, 11)
                    + np.power(n, 12)
                )
            )
            / ((-1.0 + n) * np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
            - (
                8.0
                * (
                    -16.0
                    - 32.0 * n
                    + 56.0 * np.power(n, 2)
                    + 158.0 * np.power(n, 3)
                    + 142.0 * np.power(n, 4)
                    + 95.0 * np.power(n, 5)
                    + 51.0 * np.power(n, 6)
                    + 23.0 * np.power(n, 7)
                    + 3.0 * np.power(n, 8)
                )
                * S1
            )
            / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (6.0 + 9.0 * n + 4.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0
                    + 10.0 * n
                    + 69.0 * np.power(n, 2)
                    + 48.0 * np.power(n, 3)
                    + 17.0 * np.power(n, 4)
                )
                * S2
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (-2.6666666666666665 * np.power(S1, 3) + 24.0 * S1 * S2 - 32.0 * S21)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (-14.0 + 5.0 * n + 5.0 * np.power(n, 2))
                * S3
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                32.0
                * (
                    -16.0
                    - 24.0 * n
                    - 26.0 * np.power(n, 2)
                    - 3.0 * np.power(n, 3)
                    + np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                    + np.power(n, 6)
                )
                * Sm2
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                (2.0 + n + np.power(n, 2))
                * (-256.0 * S1 * Sm2 + 256.0 * Sm21 - 128.0 * Sm3)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 2.0
        * (
            (
                -4.808227612638377
                * (10.0 + n + np.power(n, 2))
                * (18.0 + 5.0 * n + 5.0 * np.power(n, 2))
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                0.2962962962962963
                * (
                    -12096.0
                    - 47232.0 * n
                    - 79152.0 * np.power(n, 2)
                    - 82640.0 * np.power(n, 3)
                    - 33372.0 * np.power(n, 4)
                    + 43144.0 * np.power(n, 5)
                    + 28263.0 * np.power(n, 6)
                    - 83791.0 * np.power(n, 7)
                    - 204934.0 * np.power(n, 8)
                    - 246412.0 * np.power(n, 9)
                    - 171627.0 * np.power(n, 10)
                    - 50653.0 * np.power(n, 11)
                    + 22446.0 * np.power(n, 12)
                    + 30172.0 * np.power(n, 13)
                    + 13660.0 * np.power(n, 14)
                    + 3036.0 * np.power(n, 15)
                    + 276.0 * np.power(n, 16)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + 76.93164180221403 * S1
            - (
                0.8888888888888888
                * (
                    864.0
                    + 2400.0 * n
                    + 3464.0 * np.power(n, 2)
                    + 2960.0 * np.power(n, 3)
                    - 1116.0 * np.power(n, 4)
                    - 6516.0 * np.power(n, 5)
                    - 5002.0 * np.power(n, 6)
                    - 2298.0 * np.power(n, 7)
                    - 1401.0 * np.power(n, 8)
                    - 452.0 * np.power(n, 9)
                    + 80.0 * np.power(n, 10)
                    + 90.0 * np.power(n, 11)
                    + 15.0 * np.power(n, 12)
                )
                * S1
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 2)
            )
            - (
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
            - (
                4.0
                * (
                    48.0
                    + 80.0 * n
                    - 112.0 * np.power(n, 2)
                    - 204.0 * np.power(n, 3)
                    - 527.0 * np.power(n, 4)
                    - 454.0 * np.power(n, 5)
                    - 164.0 * np.power(n, 6)
                    - 14.0 * np.power(n, 7)
                    + 3.0 * np.power(n, 8)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (-26.0 + 5.0 * n + 5.0 * np.power(n, 2))
                * S3
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                32.0
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * Sm2
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            - (
                32.0
                * (
                    -32.0
                    - 12.0 * n
                    + 4.0 * np.power(n, 2)
                    + 81.0 * np.power(n, 3)
                    + 28.0 * np.power(n, 4)
                    + np.power(n, 5)
                    + 13.0 * np.power(n, 6)
                    + 10.0 * np.power(n, 7)
                    + 3.0 * np.power(n, 8)
                )
                * Sm2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                64.0
                * (
                    16.0
                    + 6.0 * n
                    + 7.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * S1
                * Sm2
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (32.0 * (2.0 + n + np.power(n, 2)) * (14.0 + n + np.power(n, 2)) * Sm21)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                np.power(2.0 + n + np.power(n, 2), 2)
                * (
                    -14.424682837915132
                    + 2.6666666666666665 * np.power(S1, 3)
                    - 40.0 * S1 * S2
                    + 32.0 * S21
                    - 32.0 * S1 * Sm2
                    - 16.0 * Sm3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (96.0 * (2.0 + n + np.power(n, 2)) * (4.0 + n + np.power(n, 2)) * Sm3)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - 4.5
        * (
            (
                76.93164180221403
                * (
                    6.0
                    + 5.0 * n
                    + 7.0 * np.power(n, 2)
                    + 4.0 * np.power(n, 3)
                    + 2.0 * np.power(n, 4)
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                0.012345679012345678
                * (
                    41472.0
                    + 222336.0 * n
                    + 559488.0 * np.power(n, 2)
                    + 895648.0 * np.power(n, 3)
                    + 1.047352e6 * np.power(n, 4)
                    + 963340.0 * np.power(n, 5)
                    + 733338.0 * np.power(n, 6)
                    + 623697.0 * np.power(n, 7)
                    + 531516.0 * np.power(n, 8)
                    + 375431.0 * np.power(n, 9)
                    + 208394.0 * np.power(n, 10)
                    + 79295.0 * np.power(n, 11)
                    + 19944.0 * np.power(n, 12)
                    + 2493.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - 76.93164180221403 * S1
            - (
                0.04938271604938271
                * (
                    -3888.0
                    - 12312.0 * n
                    - 33372.0 * np.power(n, 2)
                    - 40374.0 * np.power(n, 3)
                    + 14578.0 * np.power(n, 4)
                    + 113169.0 * np.power(n, 5)
                    + 140861.0 * np.power(n, 6)
                    + 93240.0 * np.power(n, 7)
                    + 44700.0 * np.power(n, 8)
                    + 17235.0 * np.power(n, 9)
                    + 5939.0 * np.power(n, 10)
                    + 2058.0 * np.power(n, 11)
                    + 310.0 * np.power(n, 12)
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
                1.7777777777777777
                * (
                    -24.0
                    - 152.0 * n
                    - 274.0 * np.power(n, 2)
                    - 241.0 * np.power(n, 3)
                    - 113.0 * np.power(n, 4)
                    + 9.0 * np.power(n, 5)
                    + 3.0 * np.power(n, 6)
                )
                * S2
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + S1 * (71.11111111111111 * S2 - 10.666666666666666 * S3)
            + (
                0.8888888888888888
                * (
                    54.0
                    + 33.0 * n
                    + 73.0 * np.power(n, 2)
                    + 80.0 * np.power(n, 3)
                    + 40.0 * np.power(n, 4)
                )
                * S3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                1.7777777777777777
                * (
                    360.0
                    + 108.0 * n
                    - 586.0 * np.power(n, 2)
                    - 1483.0 * np.power(n, 3)
                    - 848.0 * np.power(n, 4)
                    + 239.0 * np.power(n, 5)
                    + 691.0 * np.power(n, 6)
                    + 524.0 * np.power(n, 7)
                    + 131.0 * np.power(n, 8)
                )
                * Sm2
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
                    -108.0
                    - 36.0 * n
                    - 85.0 * np.power(n, 2)
                    - 78.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                    + 60.0 * np.power(n, 5)
                    + 20.0 * np.power(n, 6)
                )
                * S1
                * Sm2
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + 10.666666666666666 * np.power(Sm2, 2)
            + (
                (
                    -108.0
                    - 72.0 * n
                    - 121.0 * np.power(n, 2)
                    - 78.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                    + 60.0 * np.power(n, 5)
                    + 20.0 * np.power(n, 6)
                )
                * (-3.5555555555555554 * Sm21 + 1.7777777777777777 * Sm3)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    return a_gg_l0 + a_gg_l1 * L + a_gg_l2 * L**2 + a_gg_l3 * L**3
