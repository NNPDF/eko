# -*- coding: utf-8 -*-
import numba as nb
import numpy as np


@nb.njit("c16(c16,c16[:],c16[:],c16[:],c16[:],u4)", cache=True)
def A_qg_3(n, sx, smx, s3x, s4x, nf):
    r"""
    Computes the |N3LO| singlet |OME| :math:`A_{qg}^{S,(3)}(N)`.
    The expression is presented in :cite:`Bierenbaum:2009mv`

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
        A_qg_3 : complex
            :math:`A_{qg}^{S,(3)}(N)`
    """
    S1, S2, S3, S4 = sx[0], sx[1], sx[2], sx[3]
    Sm2, Sm3, Sm4 = smx[1], smx[2], smx[3]
    S21 = s3x[0]
    S31, S211 = s4x[0], s4x[1]
    return 0.3333333333333333 * nf * (
        (
            -8.547960200246003
            * (2.0 + n + np.power(n, 2))
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
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (
            0.00411522633744856
            * (
                995328.0
                + 5.612544e6 * n
                + 1.5388416e7 * np.power(n, 2)
                + 2.6395008e7 * np.power(n, 3)
                + 2.9337472e7 * np.power(n, 4)
                + 2.2643488e7 * np.power(n, 5)
                + 1.6104128e7 * np.power(n, 6)
                + 1.3846104e7 * np.power(n, 7)
                + 1.1303496e7 * np.power(n, 8)
                + 1.1536274e7 * np.power(n, 9)
                + 1.7070917e7 * np.power(n, 10)
                + 2.0248499e7 * np.power(n, 11)
                + 1.6391845e7 * np.power(n, 12)
                + 9.348807e6 * np.power(n, 13)
                + 3.812487e6 * np.power(n, 14)
                + 1.064857e6 * np.power(n, 15)
                + 180999.0 * np.power(n, 16)
                + 13923.0 * np.power(n, 17)
            )
        )
        / ((-1.0 + n) * np.power(n, 6) * np.power(1.0 + n, 6) * np.power(2.0 + n, 5))
        - (
            0.06584362139917696
            * (
                1344.0
                + 7930.0 * n
                + 14077.0 * np.power(n, 2)
                + 11200.0 * np.power(n, 3)
                + 5124.0 * np.power(n, 4)
                + 1523.0 * np.power(n, 5)
            )
            * S1
        )
        / (np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
        + (
            0.09876543209876543
            * (
                120.0
                + 748.0 * n
                + 930.0 * np.power(n, 2)
                + 481.0 * np.power(n, 3)
                + 215.0 * np.power(n, 4)
            )
            * np.power(S1, 2)
        )
        / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (
            0.2962962962962963
            * (
                40.0
                + 324.0 * n
                + 478.0 * np.power(n, 2)
                + 291.0 * np.power(n, 3)
                + 109.0 * np.power(n, 4)
            )
            * S2
        )
        / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (
            (6.0 + 29.0 * n + 13.0 * np.power(n, 2) + 10.0 * np.power(n, 3))
            * (-0.19753086419753085 * np.power(S1, 3) - 0.5925925925925926 * S1 * S2)
        )
        / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
        + (
            0.3950617283950617
            * (-6.0 + n - 16.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
            * S3
        )
        / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
        - (
            4.0
            * (2.0 + n + np.power(n, 2))
            * (
                -8.547960200246003 * S1
                - 0.037037037037037035 * np.power(S1, 4)
                - 0.2222222222222222 * np.power(S1, 2) * S2
                - 0.1111111111111111 * np.power(S2, 2)
                - 0.2962962962962963 * S1 * S3
                + 1.5555555555555556 * S4
            )
        )
        / (n * (1.0 + n) * (2.0 + n))
    ) + 0.75 * nf * (
        (68.38368160196802 * (1.0 + n + np.power(n, 2)) * (2.0 + n + np.power(n, 2)))
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (
            0.03292181069958848
            * (
                -34560.0
                - 160128.0 * n
                - 264192.0 * np.power(n, 2)
                - 217952.0 * np.power(n, 3)
                - 499824.0 * np.power(n, 4)
                - 1.907512e6 * np.power(n, 5)
                - 4.373672e6 * np.power(n, 6)
                - 6.333994e6 * np.power(n, 7)
                - 6.01512e6 * np.power(n, 8)
                - 3.525799e6 * np.power(n, 9)
                - 860568.0 * np.power(n, 10)
                + 416251.0 * np.power(n, 11)
                + 471164.0 * np.power(n, 12)
                + 194011.0 * np.power(n, 13)
                + 39780.0 * np.power(n, 14)
                + 3315.0 * np.power(n, 15)
            )
        )
        / ((-1.0 + n) * np.power(n, 5) * np.power(1.0 + n, 5) * np.power(2.0 + n, 5))
        + (
            0.06584362139917696
            * (
                864.0
                - 11264.0 * n
                - 64352.0 * np.power(n, 2)
                - 115200.0 * np.power(n, 3)
                - 69902.0 * np.power(n, 4)
                + 49344.0 * np.power(n, 5)
                + 114495.0 * np.power(n, 6)
                + 90323.0 * np.power(n, 7)
                + 40547.0 * np.power(n, 8)
                + 10557.0 * np.power(n, 9)
                + 1244.0 * np.power(n, 10)
            )
            * S1
        )
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (
            0.19753086419753085
            * (
                680.0
                + 2704.0 * n
                + 4494.0 * np.power(n, 2)
                + 3991.0 * np.power(n, 3)
                + 2148.0 * np.power(n, 4)
                + 694.0 * np.power(n, 5)
                + 103.0 * np.power(n, 6)
            )
            * np.power(S1, 2)
        )
        / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (
            0.19753086419753085
            * (
                752.0
                + 3220.0 * n
                + 5724.0 * np.power(n, 2)
                + 5776.0 * np.power(n, 3)
                + 3438.0 * np.power(n, 4)
                + 1093.0 * np.power(n, 5)
                + 139.0 * np.power(n, 6)
            )
            * S2
        )
        / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (
            (
                20.0
                + 49.0 * n
                + 41.0 * np.power(n, 2)
                + 20.0 * np.power(n, 3)
                + 5.0 * np.power(n, 4)
            )
            * (
                0.3950617283950617 * np.power(S1, 3)
                - 1.1851851851851851 * S1 * S2
                + 4.7407407407407405 * S21
            )
        )
        / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (
            0.7901234567901234
            * (
                20.0
                + 31.0 * n
                + 59.0 * np.power(n, 2)
                + 38.0 * np.power(n, 3)
                + 5.0 * np.power(n, 4)
            )
            * S3
        )
        / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        - (
            0.3950617283950617
            * (224.0 + 414.0 * n + 293.0 * np.power(n, 2) + 121.0 * np.power(n, 3))
            * Sm2
        )
        / (n * np.power(1.0 + n, 2) * (2.0 + n))
        + (4.7407407407407405 * (10.0 + 8.0 * n + 5.0 * np.power(n, 2)) * Sm3)
        / (n * (1.0 + n) * (2.0 + n))
        - (
            4.0
            * (2.0 + n + np.power(n, 2))
            * (
                8.547960200246003 * S1
                + 0.037037037037037035 * np.power(S1, 4)
                - 0.2222222222222222 * np.power(S1, 2) * S2
                + 0.1111111111111111 * np.power(S2, 2)
                - 1.7777777777777777 * S211
                + S1 * (1.7777777777777777 * S21 - 1.4814814814814814 * S3)
                + 3.5555555555555554 * S31
                + 1.5555555555555556 * S4
                + 3.5555555555555554 * Sm4
            )
        )
        / (n * (1.0 + n) * (2.0 + n))
    )
