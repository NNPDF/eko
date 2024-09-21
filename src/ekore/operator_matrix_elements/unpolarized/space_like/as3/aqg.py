"""The unpolarized, space-like |N3LO| quark-gluon |OME|."""

import numba as nb
import numpy as np

from .....harmonics import cache as c


@nb.njit(cache=True)
def A_qg(n, cache, nf, L):
    r"""Compute the |N3LO| singlet |OME| :math:`A_{qg}^{S,(3)}(N)`.

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
        :math:`A_{qg}^{S,(3)}(N)`
    """
    S1 = c.get(c.S1, cache, n)
    S2 = c.get(c.S2, cache, n)
    Sm2 = c.get(c.Sm2, cache, n, is_singlet=True)
    S3 = c.get(c.S3, cache, n)
    S21 = c.get(c.S21, cache, n)
    Sm3 = c.get(c.Sm3, cache, n, is_singlet=True)
    S4 = c.get(c.S4, cache, n)
    S31 = c.get(c.S31, cache, n)
    S211 = c.get(c.S211, cache, n)
    Sm4 = c.get(c.Sm4, cache, n, is_singlet=True)
    a_qg_l0 = 0.3333333333333333 * nf * (
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
    a_qg_l3 = (
        (29.62962962962963 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (14.222222222222221 * nf)
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (23.703703703703702 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (37.925925925925924 * nf)
        / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (16.88888888888889 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        - (4.148148148148148 * np.power(n, 3) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        - (3.5555555555555554 * np.power(n, 4) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        - (0.8888888888888888 * np.power(n, 5) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
        + (21.333333333333332 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (10.666666666666666 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (16.0 * nf) / ((-1.0 + n) * n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (10.666666666666666 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (5.333333333333333 * np.power(n, 2) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        - (1.4814814814814814 * nf * S1) / ((1.0 + n) * (2.0 + n))
        - (2.962962962962963 * nf * S1) / (n * (1.0 + n) * (2.0 + n))
        - (1.4814814814814814 * n * nf * S1) / ((1.0 + n) * (2.0 + n))
    )
    a_qg_l2 = (
        (754.9629629629629 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (170.66666666666666 * nf)
        / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (512.0 * nf)
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (808.2962962962963 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (1085.6296296296296 * nf)
        / ((-1.0 + n) * n * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (331.25925925925924 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (216.74074074074073 * np.power(n, 2) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (224.14814814814815 * np.power(n, 3) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (191.11111111111111 * np.power(n, 4) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (103.4074074074074 * np.power(n, 5) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (34.22222222222222 * np.power(n, 6) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        + (4.888888888888889 * np.power(n, 7) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
        - (674.6666666666666 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (106.66666666666667 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (448.0 * nf) / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (746.6666666666666 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (478.0 * np.power(n, 2) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (121.33333333333333 * np.power(n, 3) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (37.333333333333336 * np.power(n, 4) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (36.0 * np.power(n, 5) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (6.0 * np.power(n, 6) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (154.66666666666666 * nf * S1) / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (53.333333333333336 * nf * S1)
        / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (125.33333333333333 * n * nf * S1)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (53.333333333333336 * np.power(n, 2) * nf * S1)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (13.333333333333334 * np.power(n, 3) * nf * S1)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        - (9.481481481481481 * nf * S1) / ((1.0 + n) * (2.0 + n))
        - (7.111111111111111 * nf * S1) / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
        - (22.51851851851852 * nf * S1) / (n * (1.0 + n) * (2.0 + n))
        - (5.925925925925926 * n * nf * S1) / ((1.0 + n) * (2.0 + n))
        - (2.2222222222222223 * nf * np.power(S1, 2)) / ((1.0 + n) * (2.0 + n))
        - (4.444444444444445 * nf * np.power(S1, 2)) / (n * (1.0 + n) * (2.0 + n))
        - (2.2222222222222223 * n * nf * np.power(S1, 2)) / ((1.0 + n) * (2.0 + n))
        - (2.2222222222222223 * nf * S2) / ((1.0 + n) * (2.0 + n))
        - (4.444444444444445 * nf * S2) / (n * (1.0 + n) * (2.0 + n))
        - (2.2222222222222223 * n * nf * S2) / ((1.0 + n) * (2.0 + n))
        - (8.0 * nf * Sm2) / ((1.0 + n) * (2.0 + n))
        - (16.0 * nf * Sm2) / (n * (1.0 + n) * (2.0 + n))
        - (8.0 * n * nf * Sm2) / ((1.0 + n) * (2.0 + n))
    )
    a_qg_l1 = (
        (7208.2962962962965 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (853.3333333333334 * nf)
        / ((-1.0 + n) * np.power(n, 5) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (3640.8888888888887 * nf)
        / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (7808.0 * nf)
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (11938.765432098766 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (12726.913580246914 * nf)
        / ((-1.0 + n) * n * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (1396.8395061728395 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (1495.4074074074074 * np.power(n, 2) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (3089.037037037037 * np.power(n, 3) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (3217.8765432098767 * np.power(n, 4) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (2158.1728395061727 * np.power(n, 5) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (1036.148148148148 * np.power(n, 6) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (356.69135802469134 * np.power(n, 7) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (78.51851851851852 * np.power(n, 8) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        - (7.851851851851852 * np.power(n, 9) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 5) * np.power(2.0 + n, 4))
        + (5069.333333333333 * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (384.0 * nf)
        / ((-1.0 + n) * np.power(n, 4) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (1856.0 * nf)
        / ((-1.0 + n) * np.power(n, 3) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (3363.5555555555557 * nf)
        / ((-1.0 + n) * np.power(n, 2) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (1706.6666666666667 * nf)
        / ((-1.0 + n) * n * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        + (12603.111111111111 * n * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        + (14692.888888888889 * np.power(n, 2) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        + (9295.111111111111 * np.power(n, 3) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        + (1911.111111111111 * np.power(n, 4) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (1507.5555555555557 * np.power(n, 5) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (1289.7777777777778 * np.power(n, 6) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (396.0 * np.power(n, 7) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (44.0 * np.power(n, 8) * nf)
        / ((-1.0 + n) * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
        - (1249.7777777777778 * nf * S1) / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (302.22222222222223 * nf * S1)
        / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (2168.0 * n * nf * S1) / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (1885.7777777777778 * np.power(n, 2) * nf * S1)
        / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (936.0 * np.power(n, 3) * nf * S1)
        / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (280.44444444444446 * np.power(n, 4) * nf * S1)
        / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        - (41.77777777777778 * np.power(n, 5) * nf * S1)
        / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
        + (117.33333333333333 * nf * S1) / (np.power(1.0 + n, 2) * (2.0 + n))
        + (23.703703703703702 * nf * S1)
        / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
        + (103.50617283950618 * nf * S1) / (n * np.power(1.0 + n, 2) * (2.0 + n))
        + (50.76543209876543 * n * nf * S1) / (np.power(1.0 + n, 2) * (2.0 + n))
        + (20.34567901234568 * np.power(n, 2) * nf * S1)
        / (np.power(1.0 + n, 2) * (2.0 + n))
        + (130.66666666666666 * nf * np.power(S1, 2))
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (53.333333333333336 * nf * np.power(S1, 2))
        / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (109.33333333333333 * n * nf * np.power(S1, 2))
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (53.333333333333336 * np.power(n, 2) * nf * np.power(S1, 2))
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (13.333333333333334 * np.power(n, 3) * nf * np.power(S1, 2))
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        - (7.703703703703703 * nf * np.power(S1, 2)) / ((1.0 + n) * (2.0 + n))
        - (3.5555555555555554 * nf * np.power(S1, 2))
        / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
        - (17.185185185185187 * nf * np.power(S1, 2)) / (n * (1.0 + n) * (2.0 + n))
        - (5.925925925925926 * n * nf * np.power(S1, 2)) / ((1.0 + n) * (2.0 + n))
        - (0.7407407407407407 * nf * np.power(S1, 3)) / ((1.0 + n) * (2.0 + n))
        - (1.4814814814814814 * nf * np.power(S1, 3)) / (n * (1.0 + n) * (2.0 + n))
        - (0.7407407407407407 * n * nf * np.power(S1, 3)) / ((1.0 + n) * (2.0 + n))
        + (114.66666666666667 * nf * S2) / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (53.333333333333336 * nf * S2)
        / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (125.33333333333333 * n * nf * S2)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (69.33333333333333 * np.power(n, 2) * nf * S2)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        + (13.333333333333334 * np.power(n, 3) * nf * S2)
        / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
        - (14.814814814814815 * nf * S2) / ((1.0 + n) * (2.0 + n))
        - (3.5555555555555554 * nf * S2) / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
        - (17.185185185185187 * nf * S2) / (n * (1.0 + n) * (2.0 + n))
        - (5.925925925925926 * n * nf * S2) / ((1.0 + n) * (2.0 + n))
        + (5.777777777777778 * nf * S1 * S2) / ((1.0 + n) * (2.0 + n))
        + (11.555555555555555 * nf * S1 * S2) / (n * (1.0 + n) * (2.0 + n))
        + (5.777777777777778 * n * nf * S1 * S2) / ((1.0 + n) * (2.0 + n))
        - (16.0 * nf * S21) / ((1.0 + n) * (2.0 + n))
        - (32.0 * nf * S21) / (n * (1.0 + n) * (2.0 + n))
        - (16.0 * n * nf * S21) / ((1.0 + n) * (2.0 + n))
        - (5.037037037037037 * nf * S3) / ((1.0 + n) * (2.0 + n))
        - (10.074074074074074 * nf * S3) / (n * (1.0 + n) * (2.0 + n))
        - (5.037037037037037 * n * nf * S3) / ((1.0 + n) * (2.0 + n))
        + (42.666666666666664 * nf * Sm2) / ((1.0 + n) * (2.0 + n))
        + (53.333333333333336 * nf * Sm2) / (n * (1.0 + n) * (2.0 + n))
        + (26.666666666666668 * n * nf * Sm2) / ((1.0 + n) * (2.0 + n))
        - (16.0 * nf * Sm3) / ((1.0 + n) * (2.0 + n))
        - (32.0 * nf * Sm3) / (n * (1.0 + n) * (2.0 + n))
        - (16.0 * n * nf * Sm3) / ((1.0 + n) * (2.0 + n))
    )
    return a_qg_l0 + a_qg_l1 * L + a_qg_l2 * L**2 + a_qg_l3 * L**3
