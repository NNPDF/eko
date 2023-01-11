import numba as nb
import numpy as np


@nb.njit(cache=True)
def A_ggTF2(n, sx):
    r"""Computes the approximate incomplete part of :math:`A_{gg}^{S,(3)}(N)`
    proportional to :math:`T_{F}^2`.
    The expression is presented in  :cite:`Ablinger:2014uka` (eq 4.2).
    It contains a binomial factor which is given approximated.

    When using the code, please cite the complete list of references
    available in :mod:`ekore.matching_conditions.as3`.

    Parameters
    ----------
    n : complex
        Mellin moment
    sx : list
        harmonic sums cache

    Returns
    -------
    complex
        :math:`A_{gg,T_{F}^2}^{S,(3)}(N)`

    """
    S1 = sx[0][0]
    S2 = sx[1][0]
    S3 = sx[2][0]
    S4 = sx[3][0]
    S5 = sx[4][0]
    S21 = sx[2][1]
    # Parametrization of:
    #   4^(1-n) Binomial[2 n,n] (
    #       -7 Zeta[3]+ Sum[(4^x(x!)^2(S[1,x] x-1))/((2x)! x^3),{x,1,n}]
    #   )
    binfact = (
        0.059948671922992594 / n**6
        - 0.4031217320467463 / n**5
        + 1.403209875771271 / n**4
        - 0.544794167689904 / n**3
        + 7.76187898421785 / n**2
        - 15.649239550981758 / n
        - 0.011266509219850807 * S1
        - (1.7592804440583516 * S1) / n**4
        - (1.8954989125784512 * S1) / n**3
        + (2.332743900646393 * S1) / n**2
        - (8.046565663497601 * S1) / n
        + 0.001945690098928153 * S1**2
        - 0.00015663521937123707 * S1**3
        + 4.946093398053013 * 1e-6 * S1**4
        + 0.23528272634538147 * S2
        - 0.11909084868218014 * S3
        - 0.35585050303133825 * S4
        + 0.16105352767503742 * S5
    )
    return 0.3333333333333333 * (
        (
            0.1335618781288438
            * (
                -1472.0
                - 1472.0 * n
                - 1714.0 * np.power(n, 2)
                - 547.0 * np.power(n, 3)
                - 431.0 * np.power(n, 4)
                - 189.0 * np.power(n, 5)
                - 63.0 * np.power(n, 6)
            )
        )
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        - (
            0.00823045267489712
            * (
                18144.0
                - 21600.0 * n
                - 167688.0 * np.power(n, 2)
                - 407328.0 * np.power(n, 3)
                - 325576.0 * np.power(n, 4)
                + 89818.0 * np.power(n, 5)
                - 807075.0 * np.power(n, 6)
                - 1.672874e6 * np.power(n, 7)
                - 379547.0 * np.power(n, 8)
                + 593774.0 * np.power(n, 9)
                + 61883.0 * np.power(n, 10)
                - 152862.0 * np.power(n, 11)
                - 9409.0 * np.power(n, 12)
                + 35472.0 * np.power(n, 13)
                + 8868.0 * np.power(n, 14)
            )
        )
        / (
            (-1.0 + n)
            * np.power(n, 5)
            * np.power(1.0 + n, 5)
            * (2.0 + n)
            * (-3.0 + 2.0 * n)
            * (-1.0 + 2.0 * n)
        )
        + (
            0.5925925925925926
            * (
                -24.0
                - 56.0 * n
                - 100.0 * np.power(n, 2)
                - 129.0 * np.power(n, 3)
                - 50.0 * np.power(n, 4)
                + 3.0 * np.power(n, 5)
                + 4.0 * np.power(n, 6)
            )
            * np.power(S1, 2)
        )
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
        + (0.5925925925925926 * np.power(2.0 + n + np.power(n, 2), 2) * np.power(S1, 3))
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + 1.6449340668482262
        * (
            (
                -0.8888888888888888
                * (
                    84.0
                    + 148.0 * n
                    + 245.0 * np.power(n, 2)
                    + 282.0 * np.power(n, 3)
                    - 74.0 * np.power(n, 4)
                    - 108.0 * np.power(n, 5)
                    + 106.0 * np.power(n, 6)
                    + 132.0 * np.power(n, 7)
                    + 33.0 * np.power(n, 8)
                )
            )
            / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - (
            1.7777777777777777
            * (
                -24.0
                - 56.0 * n
                - 100.0 * np.power(n, 2)
                - 129.0 * np.power(n, 3)
                - 50.0 * np.power(n, 4)
                + 3.0 * np.power(n, 5)
                + 4.0 * np.power(n, 6)
            )
            * S2
        )
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
        + S1
        * (
            (
                -0.3950617283950617
                * (
                    216.0
                    + 1008.0 * n
                    + 2082.0 * np.power(n, 2)
                    + 2564.0 * np.power(n, 3)
                    + 2192.0 * np.power(n, 4)
                    + 2206.0 * np.power(n, 5)
                    + 1470.0 * np.power(n, 6)
                    + 388.0 * np.power(n, 7)
                    - 221.0 * np.power(n, 8)
                    + 136.0 * np.power(n, 9)
                    + 23.0 * np.power(n, 10)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * (2.0 + n)
                * (-3.0 + 2.0 * n)
                * (-1.0 + 2.0 * n)
            )
            - (5.333333333333333 * np.power(2.0 + n + np.power(n, 2), 2) * S2)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
        - (
            1.0
            * np.power(2.0 + n + np.power(n, 2), 2)
            * (-21.333333333333332 * S21 + 13.037037037037036 * S3)
        )
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        - (
            4.0
            * (
                -80.0
                - 104.0 * n
                + 44.0 * np.power(n, 2)
                + 47.0 * np.power(n, 3)
                - 53.0 * np.power(n, 4)
                + 9.0 * np.power(n, 5)
                + 9.0 * np.power(n, 6)
            )
            * binfact
        )
        / (
            3
            * (-1.0 + n)
            * n
            * np.power(1.0 + n, 2)
            * (2.0 + n)
            * (-3.0 + 2.0 * n)
            * (-1.0 + 2.0 * n)
        )
    ) + 0.75 * (
        (
            0.00027434842249657066
            * (
                181440.0
                - 518400.0 * n
                - 1.22544e6 * np.power(n, 2)
                + 2.452488e6 * np.power(n, 3)
                + 2.36045e6 * np.power(n, 4)
                - 1.1167685e7 * np.power(n, 5)
                - 2.1164117e7 * np.power(n, 6)
                - 1.4957774e7 * np.power(n, 7)
                - 710852.0 * np.power(n, 8)
                + 6.431215e6 * np.power(n, 9)
                + 4.037555e6 * np.power(n, 10)
                + 481788.0 * np.power(n, 11)
                + 149796.0 * np.power(n, 12)
            )
        )
        / (
            (-1.0 + n)
            * np.power(n, 4)
            * np.power(1.0 + n, 4)
            * (2.0 + n)
            * (-3.0 + 2.0 * n)
            * (-1.0 + 2.0 * n)
        )
        + 1.2020569031595942
        * (
            (
                -0.025925925925925925
                * (
                    -2624.0
                    - 7214.0 * n
                    - 3047.0 * np.power(n, 2)
                    + 3726.0 * np.power(n, 3)
                    + 1287.0 * np.power(n, 4)
                )
            )
            / ((-1.0 + n) * n * (1.0 + n) * (2.0 + n))
            - 41.48148148148148 * S1
        )
        + 1.6449340668482262
        * (
            (
                0.14814814814814814
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
            - 20.74074074074074 * S1
        )
        - (
            0.0021947873799725653
            * (
                -12960.0
                - 38880.0 * n
                - 7470.0 * np.power(n, 2)
                - 207066.0 * np.power(n, 3)
                - 194200.0 * np.power(n, 4)
                + 478087.0 * np.power(n, 5)
                + 196513.0 * np.power(n, 6)
                - 563492.0 * np.power(n, 7)
                - 293651.0 * np.power(n, 8)
                + 180403.0 * np.power(n, 9)
                + 96020.0 * np.power(n, 10)
            )
            * S1
        )
        / (
            (-1.0 + n)
            * np.power(n, 3)
            * np.power(1.0 + n, 3)
            * (2.0 + n)
            * (-3.0 + 2.0 * n)
            * (-1.0 + 2.0 * n)
        )
        - (
            0.02962962962962963
            * (
                -142.0
                - 629.0 * n
                - 751.0 * np.power(n, 2)
                - 223.0 * np.power(n, 3)
                + 95.0 * np.power(n, 4)
                + 70.0 * np.power(n, 5)
            )
            * np.power(S1, 2)
        )
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (
            0.02962962962962963
            * (
                -462.0
                - 1329.0 * n
                - 1621.0 * np.power(n, 2)
                - 883.0 * np.power(n, 3)
                - 135.0 * np.power(n, 4)
                + 550.0 * np.power(n, 5)
                + 220.0 * np.power(n, 6)
            )
            * S2
        )
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (
            1.0666666666666667
            * (1.0 - 7.0 * n + 4.0 * np.power(n, 2) + 4.0 * np.power(n, 3))
            * (S21 - 1.0 * S3)
        )
        / ((-1.0 + n) * n * (1.0 + n))
        - (
            (
                996.0
                + 712.0 * n
                - 1495.0 * np.power(n, 2)
                + 219.0 * np.power(n, 3)
                + 452.0 * np.power(n, 4)
                - 2094.0 * np.power(n, 5)
                + 283.0 * np.power(n, 6)
                + 539.0 * np.power(n, 7)
                + 100.0 * np.power(n, 8)
            )
            * binfact
        )
        / (
            45
            * (-1.0 + n)
            * n
            * np.power(1.0 + n, 2)
            * (2.0 + n)
            * (-3.0 + 2.0 * n)
            * (-1.0 + 2.0 * n)
        )
    )
