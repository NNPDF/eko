# -*- coding: utf-8 -*-
import numba as nb
import numpy as np

from .aggTF2 import A_ggTF2_3


@nb.njit("c16(c16,c16[:],c16[:],c16[:],c16[:],u4)", cache=True)
def A_gg_3(n, sx, smx, s3x, s4x, nf):
    r"""
    Computes the |N3LO| singlet |OME| :math:`A_{gg}^{S,(3)}(N)`.
    The experssion is presented in :cite:`Bierenbaum:2009mv`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            list S1 ... S5
        smx : numpy.ndarray
            list Sm1 ... Sm5
        s3x : numpy.ndarray
            list S21, S2m1, Sm21, Sm2m1
        s4x : numpy.ndarray
            list S31, S221, Sm22, Sm211, Sm31
        nf : int
            numeber of active flavor below the threshold

    Returns
    -------
        A_gg_3 : complex
            :math:`A_{gg}^{S,(3)}(N)`

    See Also
    --------
        A_ggTF2_3: eko.matching_conditions.n3lo.aggTF2.A_ggTF2_3
            Incomplete part proportional to :math:`T_{F}^2`.
    """
    S1, S2, S3, S4 = sx[0], sx[1], sx[2], sx[3]
    Sm2, Sm3, Sm4 = smx[1], smx[2], smx[3]
    S21, Sm21 = s3x[0], s3x[2]
    S31, S211, Sm22, Sm211, Sm31 = s4x[0], s4x[1], s4x[2], s4x[3], s4x[4]
    return (
        -0.35616500834358344
        + A_ggTF2_3(n, sx, s3x)
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
