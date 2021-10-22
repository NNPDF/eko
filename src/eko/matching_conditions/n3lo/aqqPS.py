# -*- coding: utf-8 -*-
import numba as nb
import numpy as np


@nb.njit("c16(c16,c16[:],u4)", cache=True)
def A_qqPS_3(n, sx, nf):
    r"""
    Computes the |N3LO| singlet |OME| :math:`A_{qq}^{PS,(3)}(N)`.
    The expression is presented in :cite:`Bierenbaum:2009mv`

    Parameters
    ----------
        n : complex
            Mellin moment
        sx : numpy.ndarray
            list S1 ... S5
        nf : int
            numeber of active flavor below the threshold

    Returns
    -------
        A_qqPS_3 : complex
            :math:`A_{qq}^{PS,(3)}(N)`
    """
    S1, S2, S3 = sx[0], sx[1], sx[2]
    return (
        0.3333333333333333
        * nf
        * (
            (
                -0.13168724279835392
                * (
                    -864.0
                    - 1008.0 * n
                    - 3408.0 * np.power(n, 2)
                    - 11704.0 * np.power(n, 3)
                    + 34274.0 * np.power(n, 4)
                    + 204541.0 * np.power(n, 5)
                    + 423970.0 * np.power(n, 6)
                    + 532664.0 * np.power(n, 7)
                    + 492456.0 * np.power(n, 8)
                    + 354532.0 * np.power(n, 9)
                    + 187681.0 * np.power(n, 10)
                    + 66389.0 * np.power(n, 11)
                    + 13931.0 * np.power(n, 12)
                    + 1330.0 * np.power(n, 13)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                0.3950617283950617
                * (
                    288.0
                    - 96.0 * n
                    - 928.0 * np.power(n, 2)
                    + 3272.0 * np.power(n, 3)
                    + 12030.0 * np.power(n, 4)
                    + 15396.0 * np.power(n, 5)
                    + 14606.0 * np.power(n, 6)
                    + 11454.0 * np.power(n, 7)
                    + 5937.0 * np.power(n, 8)
                    + 1744.0 * np.power(n, 9)
                    + 233.0 * np.power(n, 10)
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
                (
                    -96.0
                    - 16.0 * n
                    + 404.0 * np.power(n, 2)
                    + 452.0 * np.power(n, 3)
                    + 521.0 * np.power(n, 4)
                    + 430.0 * np.power(n, 5)
                    + 185.0 * np.power(n, 6)
                    + 40.0 * np.power(n, 7)
                )
                * (-0.5925925925925926 * np.power(S1, 2) - 0.5925925925925926 * S2)
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
                    34.19184080098401
                    + 2.962962962962963 * np.power(S1, 3)
                    + 8.88888888888889 * S1 * S2
                    + 5.925925925925926 * S3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
