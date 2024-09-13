"""The unpolarized, space-like |N3LO| quark-quark pure-singlet |OME|."""

import numba as nb
import numpy as np

from .....harmonics import cache as c


@nb.njit(cache=True)
def A_qqPS(n, cache, nf, L):
    r"""Compute the |N3LO| singlet |OME| :math:`A_{qq}^{PS,(3)}(N)`.

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
        :math:`A_{qq}^{PS,(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    S3 = c.get(c.S3, cache, n)

    a_qqPS_l0 = (
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
    a_qqPS_l3 = (1.1851851851851851 * np.power(2.0 + n + np.power(n, 2), 2) * nf) / (
        (-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n)
    )
    a_qqPS_l2 = (
        0.3333333333333333
        * nf
        * (
            (
                -3.5555555555555554
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
            + (10.666666666666666 * np.power(2.0 + n + np.power(n, 2), 2) * S1)
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    a_qqPS_l1 = (
        0.3333333333333333
        * nf
        * (
            (
                1.1851851851851851
                * (
                    144.0
                    - 48.0 * n
                    - 808.0 * np.power(n, 2)
                    + 200.0 * np.power(n, 3)
                    + 3309.0 * np.power(n, 4)
                    + 4569.0 * np.power(n, 5)
                    + 4763.0 * np.power(n, 6)
                    + 4269.0 * np.power(n, 7)
                    + 2379.0 * np.power(n, 8)
                    + 712.0 * np.power(n, 9)
                    + 95.0 * np.power(n, 10)
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
                * (-10.666666666666666 * np.power(S1, 2) - 10.666666666666666 * S2)
            )
            / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        )
    )
    return a_qqPS_l0 + a_qqPS_l1 * L + a_qqPS_l2 * L**2 + a_qqPS_l3 * L**3
