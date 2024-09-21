"""The unpolarized, space-like |N3LO| heavy-gluon |OME|."""

# pylint: skip-file
import numba as nb
import numpy as np

from .....harmonics import cache as c
from .aHg_param import a_Hg3


@nb.njit(cache=True)
def A_Hg(n, cache, nf, L):
    r"""Compute the |N3LO| singlet |OME| :math:`A_{Hg}^{S,(3)}(N)`.

    The expression is presented in :cite:`Bierenbaum:2009mv,Blumlein:2017wxd,Ablinger:2024xtt`.

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
        :math:`A_{Hg}^{S,(3)}(N)`
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
    a_Hg_l0 = (
        a_Hg3(n, cache, nf)
        + (1.0684950250307503 * (2.0 + n + np.power(n, 2)))
        / (n * (1.0 + n) * (2.0 + n))
        + 0.3333333333333333
        * nf
        * (
            (
                -1.0684950250307503
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
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.3655409037440503
                * (
                    -1728.0
                    - 5664.0 * n
                    - 9200.0 * np.power(n, 2)
                    - 15680.0 * np.power(n, 3)
                    - 20036.0 * np.power(n, 4)
                    - 17554.0 * np.power(n, 5)
                    - 6701.0 * np.power(n, 6)
                    + 5081.0 * np.power(n, 7)
                    + 9270.0 * np.power(n, 8)
                    + 6556.0 * np.power(n, 9)
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
            - (
                1.3333333333333333
                * (
                    -768.0
                    - 5248.0 * n
                    - 16064.0 * np.power(n, 2)
                    - 28256.0 * np.power(n, 3)
                    - 30384.0 * np.power(n, 4)
                    - 30808.0 * np.power(n, 5)
                    - 35844.0 * np.power(n, 6)
                    - 39994.0 * np.power(n, 7)
                    - 40778.0 * np.power(n, 8)
                    - 30218.0 * np.power(n, 9)
                    - 2639.0 * np.power(n, 10)
                    + 29583.0 * np.power(n, 11)
                    + 45159.0 * np.power(n, 12)
                    + 37119.0 * np.power(n, 13)
                    + 19019.0 * np.power(n, 14)
                    + 6055.0 * np.power(n, 15)
                    + 1099.0 * np.power(n, 16)
                    + 87.0 * np.power(n, 17)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 5)
            )
            - (
                2.9243272299524024
                * (12.0 + 28.0 * n + 11.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            + (
                5.333333333333333
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
            / (np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                2.6666666666666665
                * (
                    -4.0
                    - 18.0 * n
                    - 32.0 * np.power(n, 2)
                    - 5.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                2.6666666666666665
                * (
                    -192.0
                    - 736.0 * n
                    - 1232.0 * np.power(n, 2)
                    - 1688.0 * np.power(n, 3)
                    - 1424.0 * np.power(n, 4)
                    - 1152.0 * np.power(n, 5)
                    - 1060.0 * np.power(n, 6)
                    - 459.0 * np.power(n, 7)
                    + 74.0 * np.power(n, 8)
                    + 144.0 * np.power(n, 9)
                    + 42.0 * np.power(n, 10)
                    + 3.0 * np.power(n, 11)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                (2.0 + 3.0 * n)
                * (-1.7777777777777777 * np.power(S1, 3) - 5.333333333333333 * S1 * S2)
            )
            / (np.power(n, 2) * (2.0 + n))
            + (21.333333333333332 * (-2.0 - 3.0 * n + np.power(n, 2)) * S21)
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                1.7777777777777777
                * (
                    -144.0
                    - 200.0 * n
                    - 272.0 * np.power(n, 2)
                    - 314.0 * np.power(n, 3)
                    - 353.0 * np.power(n, 4)
                    - 44.0 * np.power(n, 5)
                    + 118.0 * np.power(n, 6)
                    + 54.0 * np.power(n, 7)
                    + 3.0 * np.power(n, 8)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -1.0684950250307503 * S1
                    - 2.193245422464302 * np.power(S1, 2)
                    - 0.1111111111111111 * np.power(S1, 4)
                    - 0.6666666666666666 * np.power(S1, 2) * S2
                    - 0.3333333333333333 * np.power(S2, 2)
                    + 10.666666666666666 * S211
                    + S1 * (-5.333333333333333 * S21 - 0.8888888888888888 * S3)
                    - 5.333333333333333 * S31
                    + 2.0 * S4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                -2.1369900500615007
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 20.0 * n
                    - 31.0 * np.power(n, 2)
                    - 16.0 * np.power(n, 3)
                    + 7.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                    + 6.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                0.3655409037440503
                * (
                    -1728.0
                    - 4992.0 * n
                    - 8944.0 * np.power(n, 2)
                    - 16288.0 * np.power(n, 3)
                    - 20572.0 * np.power(n, 4)
                    - 14684.0 * np.power(n, 5)
                    + 1193.0 * np.power(n, 6)
                    + 11479.0 * np.power(n, 7)
                    + 10350.0 * np.power(n, 8)
                    + 5378.0 * np.power(n, 9)
                    + 1701.0 * np.power(n, 10)
                    + 243.0 * np.power(n, 11)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            - (
                0.6666666666666666
                * (
                    192.0
                    + 736.0 * n
                    + 1616.0 * np.power(n, 2)
                    + 1544.0 * np.power(n, 3)
                    + 256.0 * np.power(n, 4)
                    + 1676.0 * np.power(n, 5)
                    + 3876.0 * np.power(n, 6)
                    + 905.0 * np.power(n, 7)
                    - 3313.0 * np.power(n, 8)
                    - 1207.0 * np.power(n, 9)
                    + 5375.0 * np.power(n, 10)
                    + 9235.0 * np.power(n, 11)
                    + 6877.0 * np.power(n, 12)
                    + 2567.0 * np.power(n, 13)
                    + 385.0 * np.power(n, 14)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 2)
            )
            - (
                14.621636149762011
                * (6.0 + 11.0 * n + 4.0 * np.power(n, 2) + np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            + (
                10.666666666666666
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
            / (np.power(n, 2) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                5.333333333333333
                * (
                    -4.0
                    - 18.0 * n
                    - 32.0 * np.power(n, 2)
                    - 5.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                5.333333333333333
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
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                (2.0 + 3.0 * n)
                * (-3.5555555555555554 * np.power(S1, 3) - 10.666666666666666 * S1 * S2)
            )
            / (np.power(n, 2) * (2.0 + n))
            + (42.666666666666664 * (-2.0 - 3.0 * n + np.power(n, 2)) * S21)
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                3.5555555555555554
                * (
                    -8.0
                    - 22.0 * n
                    + 43.0 * np.power(n, 2)
                    + 48.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * S3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -4.273980100123001 * S1
                    - 0.2222222222222222 * np.power(S1, 4)
                    - 1.3333333333333333 * np.power(S1, 2) * S2
                    - 0.6666666666666666 * np.power(S2, 2)
                    + 1.6449340668482262
                    * (-3.3333333333333335 * np.power(S1, 2) + 2.0 * S2)
                    + 21.333333333333332 * S211
                    + S1 * (-10.666666666666666 * S21 - 1.7777777777777777 * S3)
                    - 10.666666666666666 * S31
                    + 4.0 * S4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (-437.8296616868554 * (2.0 + n + np.power(n, 2)))
            / (n * (1.0 + n) * (2.0 + n))
            + (
                0.8013712687730628
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0
                    + 12.0 * n
                    + 165.0 * np.power(n, 2)
                    + 306.0 * np.power(n, 3)
                    + 153.0 * np.power(n, 4)
                )
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                0.8224670334241131
                * (
                    -48.0
                    - 184.0 * n
                    - 176.0 * np.power(n, 2)
                    + 1182.0 * np.power(n, 3)
                    + 4307.0 * np.power(n, 4)
                    + 6174.0 * np.power(n, 5)
                    + 5036.0 * np.power(n, 6)
                    + 2532.0 * np.power(n, 7)
                    + 633.0 * np.power(n, 8)
                )
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                1.0
                * (
                    16.0
                    + 120.0 * n
                    + 444.0 * np.power(n, 2)
                    + 1066.0 * np.power(n, 3)
                    + 1540.0 * np.power(n, 4)
                    + 246.0 * np.power(n, 5)
                    - 4163.0 * np.power(n, 6)
                    - 8462.0 * np.power(n, 7)
                    - 7605.0 * np.power(n, 8)
                    - 3148.0 * np.power(n, 9)
                    - 311.0 * np.power(n, 10)
                    + 138.0 * np.power(n, 11)
                    + 23.0 * np.power(n, 12)
                )
            )
            / (np.power(n, 6) * np.power(1.0 + n, 6) * (2.0 + n))
            - (
                13.15947253478581
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
            - (
                4.0
                * (
                    -8.0
                    - 48.0 * n
                    - 114.0 * np.power(n, 2)
                    - 90.0 * np.power(n, 3)
                    + 240.0 * np.power(n, 4)
                    + 889.0 * np.power(n, 5)
                    + 1405.0 * np.power(n, 6)
                    + 1119.0 * np.power(n, 7)
                    + 407.0 * np.power(n, 8)
                    + 62.0 * np.power(n, 9)
                    + 10.0 * np.power(n, 10)
                )
                * S1
            )
            / (np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
            - (
                6.579736267392905
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
            + (
                2.0
                * (
                    -8.0
                    - 96.0 * n
                    - 202.0 * np.power(n, 2)
                    + 208.0 * np.power(n, 3)
                    + 227.0 * np.power(n, 4)
                    + 140.0 * np.power(n, 5)
                    + 51.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                1.3333333333333333
                * (
                    -4.0
                    - 40.0 * n
                    - 111.0 * np.power(n, 2)
                    - 180.0 * np.power(n, 3)
                    - 15.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                )
                * np.power(S1, 3)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                0.3333333333333333
                * (
                    36.0
                    + 120.0 * n
                    + 139.0 * np.power(n, 2)
                    + 54.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * np.power(S1, 4)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                2.0
                * (
                    -16.0
                    - 64.0 * n
                    - 244.0 * np.power(n, 2)
                    - 636.0 * np.power(n, 3)
                    - 434.0 * np.power(n, 4)
                    + 697.0 * np.power(n, 5)
                    + 1133.0 * np.power(n, 6)
                    + 427.0 * np.power(n, 7)
                    + np.power(n, 8)
                )
                * S2
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                4.0
                * (
                    -20.0
                    - 60.0 * n
                    - 131.0 * np.power(n, 2)
                    - 119.0 * np.power(n, 3)
                    + 57.0 * np.power(n, 4)
                    + 87.0 * np.power(n, 5)
                    + 18.0 * np.power(n, 6)
                )
                * S1
                * S2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                2.0
                * (10.0 + 27.0 * n + 24.0 * np.power(n, 2) + 3.0 * np.power(n, 3))
                * np.power(S1, 2)
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                16.0
                * (-2.0 - 3.0 * n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * S21
            )
            / (np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                16.0
                * (
                    12.0
                    + 28.0 * n
                    + 19.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * S1
                * S21
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                1.3333333333333333
                * (
                    -16.0
                    - 68.0 * n
                    + 92.0 * np.power(n, 2)
                    + 399.0 * np.power(n, 3)
                    + 519.0 * np.power(n, 4)
                    + 297.0 * np.power(n, 5)
                    + 57.0 * np.power(n, 6)
                )
                * S3
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            - (
                2.6666666666666665
                * (
                    -12.0
                    - 36.0 * n
                    + 97.0 * np.power(n, 2)
                    + 102.0 * np.power(n, 3)
                    + 9.0 * np.power(n, 4)
                )
                * S1
                * S3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (
                    -6.410970150184502 * S1
                    + 13.15947253478581 * S2
                    - 1.0 * np.power(S2, 2)
                    + 32.0 * S211
                    - 16.0 * S31
                    + 6.0 * S4
                )
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (26.31894506957162 * (2.0 + n + np.power(n, 2)) * Sm2)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -3.205485075092251 * np.power(S1, 2)
                    - 0.3333333333333333 * np.power(S1, 5)
                    - 2.0 * np.power(S1, 3) * S2
                    + np.power(S1, 2) * (-16.0 * S21 - 2.6666666666666665 * S3)
                    + S1
                    * (-1.0 * np.power(S2, 2) + 32.0 * S211 - 16.0 * S31 + 6.0 * S4)
                    + 1.6449340668482262
                    * (
                        -4.0 * np.power(S1, 3)
                        + 8.0 * S1 * S2
                        + 4.0 * S3
                        + 8.0 * S1 * Sm2
                        - 8.0 * Sm21
                        + 4.0 * Sm3
                    )
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * nf
        * (
            (
                2.1369900500615007
                * (
                    8.0
                    + 12.0 * n
                    + 52.0 * np.power(n, 2)
                    - 19.0 * np.power(n, 3)
                    - 14.0 * np.power(n, 4)
                    + 9.0 * np.power(n, 5)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                0.7310818074881006
                * (
                    48.0
                    + 88.0 * n
                    - 68.0 * np.power(n, 2)
                    + 152.0 * np.power(n, 3)
                    - 357.0 * np.power(n, 4)
                    - 252.0 * np.power(n, 5)
                    + 50.0 * np.power(n, 6)
                    + 36.0 * np.power(n, 7)
                    + 15.0 * np.power(n, 8)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (
                5.333333333333333
                * (
                    64.0
                    + 448.0 * n
                    + 1392.0 * np.power(n, 2)
                    + 2400.0 * np.power(n, 3)
                    + 2268.0 * np.power(n, 4)
                    + 1500.0 * np.power(n, 5)
                    + 457.0 * np.power(n, 6)
                    - 1116.0 * np.power(n, 7)
                    - 1858.0 * np.power(n, 8)
                    - 826.0 * np.power(n, 9)
                    + 682.0 * np.power(n, 10)
                    + 1183.0 * np.power(n, 11)
                    + 765.0 * np.power(n, 12)
                    + 267.0 * np.power(n, 13)
                    + 50.0 * np.power(n, 14)
                    + 4.0 * np.power(n, 15)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            + (
                2.9243272299524024
                * (
                    20.0
                    + 28.0 * n
                    + 47.0 * np.power(n, 2)
                    + 32.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                5.333333333333333
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
            / (n * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
            + (
                2.6666666666666665
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
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                1.7777777777777777
                * (2.0 + 11.0 * n + 8.0 * np.power(n, 2) + np.power(n, 3))
                * np.power(S1, 3)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                2.6666666666666665
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
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                5.333333333333333
                * (-2.0 - 27.0 * n - 12.0 * np.power(n, 2) + 3.0 * np.power(n, 3))
                * S1
                * S2
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                14.222222222222221
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
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (
                    20.0
                    + 22.0 * n
                    + 7.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * (17.545963379714415 + 21.333333333333332 * Sm2)
            )
            / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                (-4.0 - 1.0 * n + np.power(n, 2))
                * (
                    -19.232910450553508
                    - 35.09192675942883 * S1
                    - 42.666666666666664 * S1 * Sm2
                    + 42.666666666666664 * Sm21
                    - 21.333333333333332 * Sm3
                )
            )
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    1.0684950250307503 * S1
                    + 0.1111111111111111 * np.power(S1, 4)
                    + 3.3333333333333335 * np.power(S1, 2) * S2
                    + 0.3333333333333333 * np.power(S2, 2)
                    - 2.6666666666666665 * S211
                    - 2.6666666666666665 * S31
                    + 6.0 * S4
                    + (5.333333333333333 * np.power(S1, 2) + 5.333333333333333 * S2)
                    * Sm2
                    + 1.6449340668482262
                    * (
                        1.3333333333333333 * np.power(S1, 2)
                        + 1.3333333333333333 * S2
                        + 2.6666666666666665 * Sm2
                    )
                    + S1 * (8.88888888888889 * S3 - 10.666666666666666 * Sm21)
                    + 10.666666666666666 * Sm211
                    - 5.333333333333333 * Sm22
                    + 5.333333333333333 * S1 * Sm3
                    - 5.333333333333333 * Sm31
                    + 2.6666666666666665 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * (
            (
                4.273980100123001
                * (
                    28.0
                    + 42.0 * n
                    + 92.0 * np.power(n, 2)
                    + np.power(n, 3)
                    - 4.0 * np.power(n, 4)
                    + 9.0 * np.power(n, 5)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                0.7310818074881006
                * (
                    672.0
                    + 3008.0 * n
                    + 5352.0 * np.power(n, 2)
                    + 7460.0 * np.power(n, 3)
                    + 5276.0 * np.power(n, 4)
                    + 2451.0 * np.power(n, 5)
                    + 1894.0 * np.power(n, 6)
                    + 1100.0 * np.power(n, 7)
                    + 366.0 * np.power(n, 8)
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
                0.09876543209876543
                * (
                    10368.0
                    + 59904.0 * n
                    + 165984.0 * np.power(n, 2)
                    + 328672.0 * np.power(n, 3)
                    + 592440.0 * np.power(n, 4)
                    + 1.113248e6 * np.power(n, 5)
                    + 1.704634e6 * np.power(n, 6)
                    + 1.889534e6 * np.power(n, 7)
                    + 1.57506e6 * np.power(n, 8)
                    + 1.065977e6 * np.power(n, 9)
                    + 620328.0 * np.power(n, 10)
                    + 307057.0 * np.power(n, 11)
                    + 119006.0 * np.power(n, 12)
                    + 32317.0 * np.power(n, 13)
                    + 5436.0 * np.power(n, 14)
                    + 435.0 * np.power(n, 15)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            - (
                5.848654459904805
                * (
                    20.0
                    + 43.0 * n
                    + 17.0 * np.power(n, 2)
                    + 8.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                0.19753086419753085
                * (
                    864.0
                    - 1936.0 * n
                    - 11056.0 * np.power(n, 2)
                    - 33648.0 * np.power(n, 3)
                    - 28270.0 * np.power(n, 4)
                    + 17745.0 * np.power(n, 5)
                    + 46431.0 * np.power(n, 6)
                    + 36343.0 * np.power(n, 7)
                    + 15787.0 * np.power(n, 8)
                    + 3960.0 * np.power(n, 9)
                    + 436.0 * np.power(n, 10)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                2.6666666666666665
                * (
                    -24.0
                    + 12.0 * n
                    + 14.0 * np.power(n, 2)
                    - 7.0 * np.power(n, 3)
                    + 8.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                    + 2.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                3.5555555555555554
                * (2.0 + 11.0 * n + 8.0 * np.power(n, 2) + np.power(n, 3))
                * np.power(S1, 3)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                2.6666666666666665
                * (
                    128.0
                    + 512.0 * n
                    + 904.0 * np.power(n, 2)
                    + 1172.0 * np.power(n, 3)
                    + 554.0 * np.power(n, 4)
                    + 87.0 * np.power(n, 5)
                    + 233.0 * np.power(n, 6)
                    + 193.0 * np.power(n, 7)
                    + 53.0 * np.power(n, 8)
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
            - (
                10.666666666666666
                * (-2.0 - 27.0 * n - 12.0 * np.power(n, 2) + 3.0 * np.power(n, 3))
                * S1
                * S2
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                28.444444444444443
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
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (
                    20.0
                    + 22.0 * n
                    + 7.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * (35.09192675942883 + 42.666666666666664 * Sm2)
            )
            / (np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                (-4.0 - 1.0 * n + np.power(n, 2))
                * (
                    -38.465820901107016
                    - 70.18385351885766 * S1
                    - 85.33333333333333 * S1 * Sm2
                    + 85.33333333333333 * Sm21
                    - 42.666666666666664 * Sm3
                )
            )
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    7.479465175215253 * S1
                    + 0.2222222222222222 * np.power(S1, 4)
                    + 6.666666666666667 * np.power(S1, 2) * S2
                    + 0.6666666666666666 * np.power(S2, 2)
                    - 5.333333333333333 * S211
                    - 5.333333333333333 * S31
                    + 12.0 * S4
                    + (10.666666666666666 * np.power(S1, 2) + 10.666666666666666 * S2)
                    * Sm2
                    + 1.6449340668482262
                    * (
                        3.3333333333333335 * np.power(S1, 2)
                        + 3.3333333333333335 * S2
                        + 6.666666666666667 * Sm2
                    )
                    + S1 * (17.77777777777778 * S3 - 21.333333333333332 * Sm21)
                    + 21.333333333333332 * Sm211
                    - 10.666666666666666 * Sm22
                    + 10.666666666666666 * S1 * Sm3
                    - 10.666666666666666 * Sm31
                    + 5.333333333333333 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 4.5
        * (
            (
                -0.5342475125153752
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (
                    8.0
                    + 12.0 * n
                    + 52.0 * np.power(n, 2)
                    - 19.0 * np.power(n, 3)
                    - 14.0 * np.power(n, 4)
                    + 9.0 * np.power(n, 5)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.3655409037440503
                * (
                    -3456.0
                    - 17184.0 * n
                    - 39184.0 * np.power(n, 2)
                    - 62960.0 * np.power(n, 3)
                    - 65616.0 * np.power(n, 4)
                    - 41818.0 * np.power(n, 5)
                    - 5017.0 * np.power(n, 6)
                    - 4436.0 * np.power(n, 7)
                    - 18414.0 * np.power(n, 8)
                    - 11265.0 * np.power(n, 9)
                    - 1501.0 * np.power(n, 10)
                    + 794.0 * np.power(n, 11)
                    + 420.0 * np.power(n, 12)
                    + 69.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            - (
                1.3333333333333333
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (
                    64.0
                    + 448.0 * n
                    + 1392.0 * np.power(n, 2)
                    + 2400.0 * np.power(n, 3)
                    + 2268.0 * np.power(n, 4)
                    + 1500.0 * np.power(n, 5)
                    + 457.0 * np.power(n, 6)
                    - 1116.0 * np.power(n, 7)
                    - 1858.0 * np.power(n, 8)
                    - 826.0 * np.power(n, 9)
                    + 682.0 * np.power(n, 10)
                    + 1183.0 * np.power(n, 11)
                    + 765.0 * np.power(n, 12)
                    + 267.0 * np.power(n, 13)
                    + 50.0 * np.power(n, 14)
                    + 4.0 * np.power(n, 15)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 6)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 6)
            )
            - (
                0.5342475125153752
                * (
                    -240.0
                    - 668.0 * n
                    - 356.0 * np.power(n, 2)
                    - 487.0 * np.power(n, 3)
                    - 105.0 * np.power(n, 4)
                    + 339.0 * np.power(n, 5)
                    + 77.0 * np.power(n, 6)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.7310818074881006
                * (
                    -864.0
                    - 2160.0 * n
                    - 472.0 * np.power(n, 2)
                    + 36.0 * np.power(n, 3)
                    + 4926.0 * np.power(n, 4)
                    + 3755.0 * np.power(n, 5)
                    - 1505.0 * np.power(n, 6)
                    - 334.0 * np.power(n, 7)
                    + 1124.0 * np.power(n, 8)
                    + 575.0 * np.power(n, 9)
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
                    768.0
                    + 5376.0 * n
                    + 16704.0 * np.power(n, 2)
                    + 29568.0 * np.power(n, 3)
                    + 30416.0 * np.power(n, 4)
                    + 31936.0 * np.power(n, 5)
                    + 44956.0 * np.power(n, 6)
                    + 54008.0 * np.power(n, 7)
                    + 40728.0 * np.power(n, 8)
                    + 15041.0 * np.power(n, 9)
                    + 1996.0 * np.power(n, 10)
                    + 2510.0 * np.power(n, 11)
                    + 3222.0 * np.power(n, 12)
                    + 1503.0 * np.power(n, 13)
                    + 314.0 * np.power(n, 14)
                    + 26.0 * np.power(n, 15)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            - (
                2.193245422464302
                * (
                    -48.0
                    - 116.0 * n
                    - 92.0 * np.power(n, 2)
                    - 133.0 * np.power(n, 3)
                    + 9.0 * np.power(n, 4)
                    + 81.0 * np.power(n, 5)
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
            + (
                0.6666666666666666
                * (
                    -192.0
                    - 752.0 * n
                    - 72.0 * np.power(n, 2)
                    - 6116.0 * np.power(n, 3)
                    - 9218.0 * np.power(n, 4)
                    + 1258.0 * np.power(n, 5)
                    + 9211.0 * np.power(n, 6)
                    + 6514.0 * np.power(n, 7)
                    + 2106.0 * np.power(n, 8)
                    + 392.0 * np.power(n, 9)
                    + 37.0 * np.power(n, 10)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            - (
                0.4444444444444444
                * (
                    -48.0
                    - 212.0 * n
                    - 1200.0 * np.power(n, 2)
                    - 769.0 * np.power(n, 3)
                    + 190.0 * np.power(n, 4)
                    + 208.0 * np.power(n, 5)
                    + 128.0 * np.power(n, 6)
                    + 101.0 * np.power(n, 7)
                    + 18.0 * np.power(n, 8)
                )
                * np.power(S1, 3)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                0.1111111111111111
                * (
                    -48.0
                    - 20.0 * n
                    + 292.0 * np.power(n, 2)
                    - 181.0 * np.power(n, 3)
                    - 327.0 * np.power(n, 4)
                    - 15.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * np.power(S1, 4)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.6666666666666666
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
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
                * np.power(2.0 + n, 4)
            )
            - (
                1.3333333333333333
                * (
                    384.0
                    + 1488.0 * n
                    + 1996.0 * np.power(n, 2)
                    + 2000.0 * np.power(n, 3)
                    + 359.0 * np.power(n, 4)
                    + 586.0 * np.power(n, 5)
                    + 1296.0 * np.power(n, 6)
                    + 576.0 * np.power(n, 7)
                    + 93.0 * np.power(n, 8)
                    + 6.0 * np.power(n, 9)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                0.6666666666666666
                * (-2.0 + n)
                * (
                    120.0
                    + 326.0 * n
                    + 213.0 * np.power(n, 2)
                    + 379.0 * np.power(n, 3)
                    + 347.0 * np.power(n, 4)
                    + 55.0 * np.power(n, 5)
                )
                * np.power(S1, 2)
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                3.5555555555555554
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
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
                * np.power(2.0 + n, 3)
            )
            + (
                1.7777777777777777
                * (
                    -384.0
                    - 748.0 * n
                    - 772.0 * np.power(n, 2)
                    - 401.0 * np.power(n, 3)
                    - 195.0 * np.power(n, 4)
                    + 141.0 * np.power(n, 5)
                    + 55.0 * np.power(n, 6)
                )
                * S1
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (
                    20.0
                    + 22.0 * n
                    + 7.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (-4.386490844928604 - 5.333333333333333 * Sm2)
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 4) * np.power(2.0 + n, 4))
            - (
                52.63789013914324
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (
                    96.0
                    + 232.0 * n
                    - 58.0 * np.power(n, 2)
                    - 131.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 5)
                )
                * (8.772981689857207 * S1 + 10.666666666666666 * S1 * Sm2)
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2) * np.power(2.0 + n, 3))
            + (
                (
                    -48.0
                    - 116.0 * n
                    - 44.0 * np.power(n, 2)
                    - 109.0 * np.power(n, 3)
                    - 39.0 * np.power(n, 4)
                    + 57.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * (
                    4.386490844928604 * np.power(S1, 2)
                    + 5.333333333333333 * np.power(S1, 2) * Sm2
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                (-4.0 - 1.0 * n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (
                    4.808227612638377
                    - 10.666666666666666 * Sm21
                    + 5.333333333333333 * Sm3
                )
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                (
                    -48.0
                    - 68.0 * n
                    - 24.0 * np.power(n, 2)
                    - 49.0 * np.power(n, 3)
                    + 34.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * (
                    4.808227612638377 * S1
                    - 10.666666666666666 * S1 * Sm21
                    + 5.333333333333333 * S1 * Sm3
                )
            )
            / ((-1.0 + n) * np.power(n, 2) * (1.0 + n) * np.power(2.0 + n, 2))
            + (
                (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (
                    0.3333333333333333 * np.power(S2, 2)
                    - 2.6666666666666665 * S211
                    - 2.6666666666666665 * S31
                    + 6.0 * S4
                    + 5.333333333333333 * S2 * Sm2
                    + 1.6449340668482262
                    * (2.6666666666666665 * S2 + 2.6666666666666665 * Sm2)
                    + 10.666666666666666 * Sm211
                    - 5.333333333333333 * Sm22
                    - 5.333333333333333 * Sm31
                    + 2.6666666666666665 * Sm4
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -3.205485075092251 * np.power(S1, 2)
                    - 0.3333333333333333 * np.power(S1, 5)
                    - 10.0 * np.power(S1, 3) * S2
                    + (-16.0 * np.power(S1, 3) - 16.0 * S1 * S2) * Sm2
                    + np.power(S1, 2) * (-26.666666666666668 * S3 + 32.0 * Sm21)
                    + 1.6449340668482262
                    * (
                        -4.0 * np.power(S1, 3)
                        + 3.6666666666666665 * S2
                        - 8.0 * S1 * S2
                        - 2.0 * S3
                        - 12.0 * S1 * Sm2
                        + 4.0 * Sm21
                        - 2.0 * Sm3
                    )
                    - 16.0 * np.power(S1, 2) * Sm3
                    + S1
                    * (
                        -1.0 * np.power(S2, 2)
                        + 8.0 * S211
                        + 8.0 * S31
                        - 18.0 * S4
                        - 32.0 * Sm211
                        + 16.0 * Sm22
                        + 16.0 * Sm31
                    )
                    - 8.0 * S1 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 2.0
        * (
            (218.9148308434277 * (2.0 + n + np.power(n, 2)))
            / (n * (1.0 + n) * (2.0 + n))
            - (
                0.2671237562576876
                * (
                    192.0
                    + 664.0 * n
                    - 404.0 * np.power(n, 2)
                    - 2554.0 * np.power(n, 3)
                    - 681.0 * np.power(n, 4)
                    + 1692.0 * np.power(n, 5)
                    + 3002.0 * np.power(n, 6)
                    + 2190.0 * np.power(n, 7)
                    + 507.0 * np.power(n, 8)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.09138522593601257
                * (
                    1776.0
                    + 9488.0 * n
                    + 25144.0 * np.power(n, 2)
                    + 44064.0 * np.power(n, 3)
                    + 55339.0 * np.power(n, 4)
                    + 37623.0 * np.power(n, 5)
                    + 21430.0 * np.power(n, 6)
                    + 15070.0 * np.power(n, 7)
                    + 5751.0 * np.power(n, 8)
                    + 891.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 2)
            )
            + (
                0.3333333333333333
                * (
                    1408.0
                    - 4480.0 * n
                    - 64672.0 * np.power(n, 2)
                    - 200160.0 * np.power(n, 3)
                    - 261272.0 * np.power(n, 4)
                    - 73752.0 * np.power(n, 5)
                    + 207634.0 * np.power(n, 6)
                    + 337718.0 * np.power(n, 7)
                    + 425270.0 * np.power(n, 8)
                    + 712841.0 * np.power(n, 9)
                    + 1.086519e6 * np.power(n, 10)
                    + 1.160715e6 * np.power(n, 11)
                    + 831483.0 * np.power(n, 12)
                    + 394315.0 * np.power(n, 13)
                    + 119399.0 * np.power(n, 14)
                    + 20963.0 * np.power(n, 15)
                    + 1623.0 * np.power(n, 16)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 6)
                * np.power(2.0 + n, 5)
            )
            - (
                0.5342475125153752
                * (
                    -312.0
                    - 556.0 * n
                    - 1054.0 * np.power(n, 2)
                    + 259.0 * np.power(n, 3)
                    + 351.0 * np.power(n, 4)
                    + 93.0 * np.power(n, 5)
                    + 67.0 * np.power(n, 6)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                0.7310818074881006
                * (
                    144.0
                    - 48.0 * n
                    - 1096.0 * np.power(n, 2)
                    + 184.0 * np.power(n, 3)
                    - 381.0 * np.power(n, 4)
                    + 358.0 * np.power(n, 5)
                    + 924.0 * np.power(n, 6)
                    + 370.0 * np.power(n, 7)
                    + 121.0 * np.power(n, 8)
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
                2.6666666666666665
                * (
                    576.0
                    + 4032.0 * n
                    + 10416.0 * np.power(n, 2)
                    + 7584.0 * np.power(n, 3)
                    - 16628.0 * np.power(n, 4)
                    - 40468.0 * np.power(n, 5)
                    - 40915.0 * np.power(n, 6)
                    - 33352.0 * np.power(n, 7)
                    - 27541.0 * np.power(n, 8)
                    - 11753.0 * np.power(n, 9)
                    + 6624.0 * np.power(n, 10)
                    + 11508.0 * np.power(n, 11)
                    + 6497.0 * np.power(n, 12)
                    + 1953.0 * np.power(n, 13)
                    + 335.0 * np.power(n, 14)
                    + 28.0 * np.power(n, 15)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            - (
                4.386490844928604
                * (
                    12.0
                    + 20.0 * n
                    - 97.0 * np.power(n, 2)
                    - 44.0 * np.power(n, 3)
                    - 39.0 * np.power(n, 4)
                    - 6.0 * np.power(n, 5)
                    + 10.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                1.3333333333333333
                * (
                    -192.0
                    - 416.0 * n
                    - 1712.0 * np.power(n, 2)
                    - 10832.0 * np.power(n, 3)
                    - 26920.0 * np.power(n, 4)
                    - 23342.0 * np.power(n, 5)
                    - 282.0 * np.power(n, 6)
                    + 12320.0 * np.power(n, 7)
                    + 9245.0 * np.power(n, 8)
                    + 3599.0 * np.power(n, 9)
                    + 853.0 * np.power(n, 10)
                    + 95.0 * np.power(n, 11)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.8888888888888888
                * (
                    -48.0
                    - 128.0 * n
                    + 84.0 * np.power(n, 2)
                    + 300.0 * np.power(n, 3)
                    - 78.0 * np.power(n, 4)
                    - 1251.0 * np.power(n, 5)
                    - 1116.0 * np.power(n, 6)
                    - 115.0 * np.power(n, 7)
                    + 156.0 * np.power(n, 8)
                    + 36.0 * np.power(n, 9)
                )
                * np.power(S1, 3)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.2222222222222222
                * (
                    84.0
                    + 296.0 * n
                    + 329.0 * np.power(n, 2)
                    - 317.0 * np.power(n, 3)
                    - 444.0 * np.power(n, 4)
                    - 93.0 * np.power(n, 5)
                    + np.power(n, 6)
                )
                * np.power(S1, 4)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                6.579736267392905
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0
                    + 4.0 * n
                    + 7.0 * np.power(n, 2)
                    + 6.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                1.3333333333333333
                * (
                    112.0
                    + 992.0 * n
                    + 2888.0 * np.power(n, 2)
                    + 5000.0 * np.power(n, 3)
                    + 8997.0 * np.power(n, 4)
                    + 13213.0 * np.power(n, 5)
                    + 12399.0 * np.power(n, 6)
                    + 7171.0 * np.power(n, 7)
                    + 2448.0 * np.power(n, 8)
                    + 456.0 * np.power(n, 9)
                    + 36.0 * np.power(n, 10)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                2.6666666666666665
                * (
                    240.0
                    + 736.0 * n
                    + 996.0 * np.power(n, 2)
                    + 588.0 * np.power(n, 3)
                    - 1116.0 * np.power(n, 4)
                    - 933.0 * np.power(n, 5)
                    + 1080.0 * np.power(n, 6)
                    + 1409.0 * np.power(n, 7)
                    + 534.0 * np.power(n, 8)
                    + 66.0 * np.power(n, 9)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                1.3333333333333333
                * (
                    -84.0
                    - 200.0 * n
                    - 389.0 * np.power(n, 2)
                    + 359.0 * np.power(n, 3)
                    + 390.0 * np.power(n, 4)
                    + 51.0 * np.power(n, 5)
                    + 17.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.6666666666666666
                * (2.0 + n + np.power(n, 2))
                * (
                    -6.0
                    - 17.0 * n
                    - 16.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * np.power(S2, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (-2.0 - 3.0 * n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S21
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (
                    -20.0
                    - 176.0 * n
                    - 145.0 * np.power(n, 2)
                    - 3.0 * np.power(n, 3)
                    + 45.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * S1
                * S21
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -84.0
                    - 172.0 * n
                    - 137.0 * np.power(n, 2)
                    + 70.0 * np.power(n, 3)
                    + 35.0 * np.power(n, 4)
                )
                * S211
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                0.4444444444444444
                * (
                    -96.0
                    + 128.0 * n
                    - 1972.0 * np.power(n, 2)
                    - 5992.0 * np.power(n, 3)
                    - 6565.0 * np.power(n, 4)
                    - 1378.0 * np.power(n, 5)
                    + 2360.0 * np.power(n, 6)
                    + 1674.0 * np.power(n, 7)
                    + 321.0 * np.power(n, 8)
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
                0.8888888888888888
                * (
                    192.0
                    + 308.0 * n
                    - 712.0 * np.power(n, 2)
                    + 229.0 * np.power(n, 3)
                    + 1311.0 * np.power(n, 4)
                    + 591.0 * np.power(n, 5)
                    + 97.0 * np.power(n, 6)
                )
                * S1
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -60.0
                    - 104.0 * n
                    - 73.0 * np.power(n, 2)
                    + 62.0 * np.power(n, 3)
                    + 31.0 * np.power(n, 4)
                )
                * S31
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -30.0
                    - 41.0 * n
                    - 22.0 * np.power(n, 2)
                    + 38.0 * np.power(n, 3)
                    + 19.0 * np.power(n, 4)
                )
                * S4
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (13.15947253478581 * (2.0 + n + np.power(n, 2)) * Sm2)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                (
                    -40.0
                    - 200.0 * n
                    - 404.0 * np.power(n, 2)
                    - 319.0 * np.power(n, 3)
                    - 65.0 * np.power(n, 4)
                    + 27.0 * np.power(n, 5)
                    + 9.0 * np.power(n, 6)
                )
                * (13.15947253478581 + 16.0 * Sm2)
            )
            / (n * np.power(1.0 + n, 4) * np.power(2.0 + n, 3))
            + (
                (
                    32.0
                    + 172.0 * n
                    + 256.0 * np.power(n, 2)
                    + 223.0 * np.power(n, 3)
                    + 136.0 * np.power(n, 4)
                    + 47.0 * np.power(n, 5)
                    + 6.0 * np.power(n, 6)
                )
                * (26.31894506957162 * S1 + 32.0 * S1 * Sm2)
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                (
                    8.0
                    + 20.0 * n
                    + 62.0 * np.power(n, 2)
                    + 31.0 * np.power(n, 3)
                    + 4.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * (13.15947253478581 * np.power(S1, 2) + 16.0 * np.power(S1, 2) * Sm2)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                (
                    16.0
                    + 58.0 * n
                    + 77.0 * np.power(n, 2)
                    + 66.0 * np.power(n, 3)
                    + 33.0 * np.power(n, 4)
                    + 6.0 * np.power(n, 5)
                )
                * (14.424682837915132 - 32.0 * Sm21 + 16.0 * Sm3)
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            + (
                (
                    8.0
                    + 20.0 * n
                    + 46.0 * np.power(n, 2)
                    + 27.0 * np.power(n, 3)
                    + 8.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * (14.424682837915132 * S1 - 32.0 * S1 * Sm21 + 16.0 * S1 * Sm3)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (
                    16.0 * S2 * Sm2
                    + 1.6449340668482262 * (8.0 * S2 + 8.0 * Sm2)
                    + 32.0 * Sm211
                    - 16.0 * Sm22
                    - 16.0 * Sm31
                    + 8.0 * Sm4
                )
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    6.410970150184502 * np.power(S1, 2)
                    + 0.6666666666666666 * np.power(S1, 5)
                    + 12.0 * np.power(S1, 3) * S2
                    + (32.0 + 16.0 * np.power(S1, 3) + 16.0 * S1 * S2) * Sm2
                    + np.power(S1, 2)
                    * (16.0 * S21 + 29.333333333333332 * S3 - 32.0 * Sm21)
                    + 1.6449340668482262
                    * (
                        8.0 * np.power(S1, 3)
                        - 2.0 * S3
                        + 4.0 * S1 * Sm2
                        + 4.0 * Sm21
                        - 2.0 * Sm3
                    )
                    + 16.0 * np.power(S1, 2) * Sm3
                    + S1
                    * (
                        2.0 * np.power(S2, 2)
                        - 40.0 * S211
                        + 8.0 * S31
                        + 12.0 * S4
                        + 32.0 * Sm211
                        - 16.0 * Sm22
                        - 16.0 * Sm31
                    )
                    + 8.0 * S1 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    a_Hg_l3 = (
        (0.8888888888888888 * (2.0 + n + np.power(n, 2))) / (n * (1.0 + n) * (2.0 + n))
        - 0.3333333333333333
        * nf
        * (
            (
                0.8888888888888888
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
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (3.5555555555555554 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * nf
        * (
            (
                -7.111111111111111
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (3.5555555555555554 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.3333333333333333
        * (
            (
                -1.7777777777777777
                * (2.0 + n + np.power(n, 2))
                * (
                    -12.0
                    - 20.0 * n
                    - 31.0 * np.power(n, 2)
                    - 16.0 * np.power(n, 3)
                    + 7.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                    + 6.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            + (14.222222222222221 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * (
            (
                -49.77777777777778
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (24.88888888888889 * (2.0 + n + np.power(n, 2)) * S1)
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (
                -0.6666666666666666
                * (2.0 + n + np.power(n, 2))
                * np.power(2.0 + 3.0 * n + 3.0 * np.power(n, 2), 2)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (10.666666666666666 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 4.5
        * (
            (
                1.7777777777777777
                * (1.0 + n + np.power(n, 2))
                * (2.0 + n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
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
                * (2.0 + n + np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (10.666666666666666 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 2.0
        * (
            (
                -0.2222222222222222
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.8888888888888888
                * (2.0 + n + np.power(n, 2))
                * (6.0 + n + np.power(n, 2))
                * (4.0 + 7.0 * n + 7.0 * np.power(n, 2))
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (21.333333333333332 * (2.0 + n + np.power(n, 2)) * np.power(S1, 2))
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    a_Hg_l2 = (
        0.3333333333333333
        * (
            (
                -0.4444444444444444
                * (
                    -1152.0
                    - 3648.0 * n
                    - 6640.0 * np.power(n, 2)
                    - 11680.0 * np.power(n, 3)
                    - 13912.0 * np.power(n, 4)
                    - 9464.0 * np.power(n, 5)
                    + 383.0 * np.power(n, 6)
                    + 6505.0 * np.power(n, 7)
                    + 5730.0 * np.power(n, 8)
                    + 2894.0 * np.power(n, 9)
                    + 903.0 * np.power(n, 10)
                    + 129.0 * np.power(n, 11)
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
                * (18.0 + 37.0 * n + 14.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (4.0 * np.power(S1, 2) - 1.3333333333333333 * S2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.3333333333333333
        * nf
        * (
            (
                -0.4444444444444444
                * (
                    1152.0
                    + 3456.0 * n
                    + 5456.0 * np.power(n, 2)
                    + 7328.0 * np.power(n, 3)
                    + 5096.0 * np.power(n, 4)
                    + 2236.0 * np.power(n, 5)
                    + 1463.0 * np.power(n, 6)
                    + 1513.0 * np.power(n, 7)
                    + 1290.0 * np.power(n, 8)
                    + 698.0 * np.power(n, 9)
                    + 231.0 * np.power(n, 10)
                    + 33.0 * np.power(n, 11)
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
                * (6.0 + 19.0 * n + 8.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * S1
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (1.3333333333333333 * np.power(S1, 2) + 1.3333333333333333 * S2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * (
            (
                -0.8888888888888888
                * (
                    384.0
                    + 1696.0 * n
                    + 2928.0 * np.power(n, 2)
                    + 3484.0 * np.power(n, 3)
                    + 2740.0 * np.power(n, 4)
                    + 1731.0 * np.power(n, 5)
                    + 1262.0 * np.power(n, 6)
                    + 724.0 * np.power(n, 7)
                    + 270.0 * np.power(n, 8)
                    + 45.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                3.5555555555555554
                * (
                    20.0
                    - 14.0 * n
                    - 1.0 * np.power(n, 2)
                    + 20.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (-4.0 * np.power(S1, 2) - 4.0 * S2 - 8.0 * Sm2)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.75
        * nf
        * (
            (
                -0.8888888888888888
                * (
                    -160.0
                    - 672.0 * n
                    - 1012.0 * np.power(n, 2)
                    - 1120.0 * np.power(n, 3)
                    - 717.0 * np.power(n, 4)
                    - 182.0 * np.power(n, 5)
                    + 56.0 * np.power(n, 6)
                    + 54.0 * np.power(n, 7)
                    + 9.0 * np.power(n, 8)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                3.5555555555555554
                * (
                    20.0
                    + 58.0 * n
                    + 47.0 * np.power(n, 2)
                    + 20.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -1.3333333333333333 * np.power(S1, 2)
                    - 1.3333333333333333 * S2
                    - 2.6666666666666665 * Sm2
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 0.8888888888888888
        * (
            (
                32.0
                + 128.0 * n
                + 292.0 * np.power(n, 2)
                + 528.0 * np.power(n, 3)
                + 685.0 * np.power(n, 4)
                + 636.0 * np.power(n, 5)
                + 350.0 * np.power(n, 6)
                + 132.0 * np.power(n, 7)
                + 33.0 * np.power(n, 8)
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                8.0
                * (
                    12.0
                    + 34.0 * n
                    + 60.0 * np.power(n, 2)
                    + 84.0 * np.power(n, 3)
                    + 51.0 * np.power(n, 4)
                    + 18.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
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
            - (
                12.0
                * (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (32.0 * (2.0 + n + np.power(n, 2)) * Sm2)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0 * np.power(S1, 3)
                    - 12.0 * S1 * S2
                    - 8.0 * S3
                    - 16.0 * S1 * Sm2
                    + 16.0 * Sm21
                    - 8.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 2.0
        * (
            (
                0.1111111111111111
                * (
                    2304.0
                    - 2864.0 * n
                    - 16272.0 * np.power(n, 2)
                    - 24608.0 * np.power(n, 3)
                    - 12692.0 * np.power(n, 4)
                    + 6675.0 * np.power(n, 5)
                    + 12206.0 * np.power(n, 6)
                    + 7636.0 * np.power(n, 7)
                    + 2934.0 * np.power(n, 8)
                    + 489.0 * np.power(n, 9)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                0.8888888888888888
                * (
                    144.0
                    + 24.0 * n
                    - 668.0 * np.power(n, 2)
                    - 314.0 * np.power(n, 3)
                    - 2.0 * np.power(n, 4)
                    + 119.0 * np.power(n, 5)
                    + 613.0 * np.power(n, 6)
                    + 635.0 * np.power(n, 7)
                    + 273.0 * np.power(n, 8)
                    + 40.0 * np.power(n, 9)
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
            + (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    18.0
                    + 7.0 * n
                    + 8.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -8.0 * np.power(S1, 3)
                    + 4.0 * S3
                    + 6.0 * Sm2
                    - 8.0 * Sm21
                    + 4.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        + 4.5
        * (
            (
                1.7777777777777777
                * (
                    -576.0
                    - 2592.0 * n
                    - 5296.0 * np.power(n, 2)
                    - 6992.0 * np.power(n, 3)
                    - 6012.0 * np.power(n, 4)
                    - 4462.0 * np.power(n, 5)
                    - 3841.0 * np.power(n, 6)
                    - 4619.0 * np.power(n, 7)
                    - 4428.0 * np.power(n, 8)
                    - 2325.0 * np.power(n, 9)
                    - 511.0 * np.power(n, 10)
                    + 101.0 * np.power(n, 11)
                    + 72.0 * np.power(n, 12)
                    + 9.0 * np.power(n, 13)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.8888888888888888
                * (
                    -576.0
                    - 1296.0 * n
                    + 968.0 * np.power(n, 2)
                    + 2568.0 * np.power(n, 3)
                    + 2238.0 * np.power(n, 4)
                    - 325.0 * np.power(n, 5)
                    - 521.0 * np.power(n, 6)
                    + 788.0 * np.power(n, 7)
                    + 830.0 * np.power(n, 8)
                    + 425.0 * np.power(n, 9)
                    + 85.0 * np.power(n, 10)
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
            - (
                1.3333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    -72.0
                    - 94.0 * n
                    - 83.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                2.6666666666666665
                * (2.0 + n + np.power(n, 2))
                * (
                    -48.0
                    - 70.0 * n
                    - 59.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    4.0 * np.power(S1, 3)
                    + 12.0 * S1 * S2
                    + 4.0 * S3
                    + 16.0 * S1 * Sm2
                    - 8.0 * Sm21
                    + 4.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    a_Hg_l1 = (
        0.3333333333333333
        * (
            (
                -0.07407407407407407
                * (
                    -34560.0
                    - 144000.0 * n
                    - 299712.0 * np.power(n, 2)
                    - 453440.0 * np.power(n, 3)
                    - 534656.0 * np.power(n, 4)
                    - 492936.0 * np.power(n, 5)
                    - 435356.0 * np.power(n, 6)
                    - 228072.0 * np.power(n, 7)
                    + 154773.0 * np.power(n, 8)
                    + 398930.0 * np.power(n, 9)
                    + 371423.0 * np.power(n, 10)
                    + 214620.0 * np.power(n, 11)
                    + 80795.0 * np.power(n, 12)
                    + 18018.0 * np.power(n, 13)
                    + 1773.0 * np.power(n, 14)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    192.0
                    + 592.0 * n
                    + 786.0 * np.power(n, 2)
                    + 163.0 * np.power(n, 3)
                    + 29.0 * np.power(n, 4)
                )
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                1.7777777777777777
                * (18.0 + 59.0 * n + 31.0 * np.power(n, 2) + 10.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (
                5.333333333333333
                * (
                    -2.0
                    - 5.0 * n
                    + 42.0 * np.power(n, 2)
                    + 39.0 * np.power(n, 3)
                    + 2.0 * np.power(n, 4)
                )
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (1.3333333333333333 * np.power(S1, 3) + 4.0 * S1 * S2 - 8.0 * S3)
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.3333333333333333
        * nf
        * (
            (
                0.07407407407407407
                * (
                    -34560.0
                    - 158976.0 * n
                    - 355584.0 * np.power(n, 2)
                    - 562304.0 * np.power(n, 3)
                    - 735104.0 * np.power(n, 4)
                    - 694320.0 * np.power(n, 5)
                    - 490544.0 * np.power(n, 6)
                    - 207408.0 * np.power(n, 7)
                    + 82971.0 * np.power(n, 8)
                    + 205070.0 * np.power(n, 9)
                    + 159437.0 * np.power(n, 10)
                    + 77604.0 * np.power(n, 11)
                    + 25877.0 * np.power(n, 12)
                    + 5454.0 * np.power(n, 13)
                    + 531.0 * np.power(n, 14)
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
                * (
                    78.0
                    + 251.0 * n
                    + 303.0 * np.power(n, 2)
                    + 77.0 * np.power(n, 3)
                    + 19.0 * np.power(n, 4)
                )
                * S1
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                3.5555555555555554
                * (6.0 + 22.0 * n + 11.0 * np.power(n, 2) + 5.0 * np.power(n, 3))
                * np.power(S1, 2)
            )
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            + (
                5.333333333333333
                * (
                    -48.0
                    - 72.0 * n
                    - 108.0 * np.power(n, 2)
                    - 128.0 * np.power(n, 3)
                    - 123.0 * np.power(n, 4)
                    + 8.0 * np.power(n, 5)
                    + 62.0 * np.power(n, 6)
                    + 24.0 * np.power(n, 7)
                    + np.power(n, 8)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    0.8888888888888888 * np.power(S1, 3)
                    + 2.6666666666666665 * S1 * S2
                    - 6.222222222222222 * S3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * (
            (
                0.5925925925925926
                * (
                    288.0
                    + 1008.0 * n
                    + 80.0 * np.power(n, 2)
                    - 4560.0 * np.power(n, 3)
                    - 19122.0 * np.power(n, 4)
                    - 34963.0 * np.power(n, 5)
                    - 37157.0 * np.power(n, 6)
                    - 21724.0 * np.power(n, 7)
                    - 2662.0 * np.power(n, 8)
                    + 5168.0 * np.power(n, 9)
                    + 3634.0 * np.power(n, 10)
                    + 1035.0 * np.power(n, 11)
                    + 111.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    536.0
                    + 1516.0 * n
                    + 3186.0 * np.power(n, 2)
                    + 3271.0 * np.power(n, 3)
                    + 1692.0 * np.power(n, 4)
                    + 487.0 * np.power(n, 5)
                    + 76.0 * np.power(n, 6)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            - (
                3.5555555555555554
                * (
                    26.0
                    + 82.0 * n
                    + 65.0 * np.power(n, 2)
                    + 23.0 * np.power(n, 3)
                    + 5.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                3.5555555555555554
                * (
                    48.0
                    + 34.0 * n
                    + 40.0 * np.power(n, 2)
                    + 35.0 * np.power(n, 3)
                    + 84.0 * np.power(n, 4)
                    + 42.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (42.666666666666664 * (-4.0 - 1.0 * n + np.power(n, 2)) * Sm2)
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (7.111111111111111 * (10.0 + 8.0 * n + 5.0 * np.power(n, 2)) * Sm2)
            / (n * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -1.3333333333333333 * np.power(S1, 3)
                    - 6.666666666666667 * S1 * S2
                    - 5.333333333333333 * S21
                    - 8.0 * S3
                    - 10.666666666666666 * S1 * Sm2
                    + 10.666666666666666 * Sm21
                    - 10.666666666666666 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.75
        * nf
        * (
            (
                0.2962962962962963
                * (
                    576.0
                    + 4128.0 * n
                    + 12416.0 * np.power(n, 2)
                    + 22080.0 * np.power(n, 3)
                    + 16644.0 * np.power(n, 4)
                    - 2110.0 * np.power(n, 5)
                    - 15710.0 * np.power(n, 6)
                    - 8917.0 * np.power(n, 7)
                    + 7139.0 * np.power(n, 8)
                    + 11990.0 * np.power(n, 9)
                    + 6742.0 * np.power(n, 10)
                    + 1845.0 * np.power(n, 11)
                    + 201.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 4)
            )
            + (
                0.5925925925925926
                * (
                    232.0
                    + 776.0 * n
                    + 1878.0 * np.power(n, 2)
                    + 1820.0 * np.power(n, 3)
                    + 777.0 * np.power(n, 4)
                    + 176.0 * np.power(n, 5)
                    + 29.0 * np.power(n, 6)
                )
                * S1
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            - (
                1.7777777777777777
                * (
                    46.0
                    + 131.0 * n
                    + 106.0 * np.power(n, 2)
                    + 43.0 * np.power(n, 3)
                    + 10.0 * np.power(n, 4)
                )
                * np.power(S1, 2)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (
                1.7777777777777777
                * (
                    48.0
                    + 14.0 * n
                    + 17.0 * np.power(n, 2)
                    + 31.0 * np.power(n, 3)
                    + 105.0 * np.power(n, 4)
                    + 63.0 * np.power(n, 5)
                    + 10.0 * np.power(n, 6)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (21.333333333333332 * (-4.0 - 1.0 * n + np.power(n, 2)) * Sm2)
            / (np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            - (7.111111111111111 * (10.0 + 8.0 * n + 5.0 * np.power(n, 2)) * Sm2)
            / (n * (1.0 + n) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -0.8888888888888888 * np.power(S1, 3)
                    - 2.6666666666666665 * S1 * S2
                    - 5.333333333333333 * S21
                    - 4.444444444444445 * S3
                    - 5.333333333333333 * S1 * Sm2
                    + 5.333333333333333 * Sm21
                    - 8.0 * Sm3
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 2.0
        * (
            (
                -7.212341418957566
                * (
                    -248.0
                    + 12.0 * n
                    - 214.0 * np.power(n, 2)
                    - 407.0 * np.power(n, 3)
                    - 91.0 * np.power(n, 4)
                    + 135.0 * np.power(n, 5)
                    + 45.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.018518518518518517
                * (
                    27648.0
                    + 143424.0 * n
                    + 372192.0 * np.power(n, 2)
                    + 664624.0 * np.power(n, 3)
                    + 606016.0 * np.power(n, 4)
                    + 189820.0 * np.power(n, 5)
                    + 204230.0 * np.power(n, 6)
                    + 650149.0 * np.power(n, 7)
                    + 820775.0 * np.power(n, 8)
                    + 532170.0 * np.power(n, 9)
                    + 206656.0 * np.power(n, 10)
                    + 53973.0 * np.power(n, 11)
                    + 7299.0 * np.power(n, 12)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 5)
                * np.power(1.0 + n, 4)
                * np.power(2.0 + n, 3)
            )
            + (
                0.2962962962962963
                * (
                    1944.0
                    + 6156.0 * n
                    + 7122.0 * np.power(n, 2)
                    + 13.0 * np.power(n, 3)
                    - 2242.0 * np.power(n, 4)
                    + 4008.0 * np.power(n, 5)
                    + 6764.0 * np.power(n, 6)
                    + 4206.0 * np.power(n, 7)
                    + 1586.0 * np.power(n, 8)
                    + 251.0 * np.power(n, 9)
                )
                * S1
            )
            / (
                (-1.0 + n)
                * np.power(n, 4)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 2)
            )
            - (
                0.4444444444444444
                * (
                    864.0
                    + 2712.0 * n
                    + 3116.0 * np.power(n, 2)
                    + 4106.0 * np.power(n, 3)
                    + 2003.0 * np.power(n, 4)
                    - 1202.0 * np.power(n, 5)
                    - 592.0 * np.power(n, 6)
                    + 568.0 * np.power(n, 7)
                    + 441.0 * np.power(n, 8)
                    + 80.0 * np.power(n, 9)
                )
                * np.power(S1, 2)
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            + (
                0.8888888888888888
                * (
                    -144.0
                    - 496.0 * n
                    - 866.0 * np.power(n, 2)
                    - 499.0 * np.power(n, 3)
                    - 46.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * np.power(S1, 3)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                1.3333333333333333
                * (
                    480.0
                    + 1944.0 * n
                    + 4020.0 * np.power(n, 2)
                    + 6446.0 * np.power(n, 3)
                    + 7513.0 * np.power(n, 4)
                    + 4345.0 * np.power(n, 5)
                    + 462.0 * np.power(n, 6)
                    - 424.0 * np.power(n, 7)
                    - 43.0 * np.power(n, 8)
                    + 25.0 * np.power(n, 9)
                )
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                2.6666666666666665
                * (3.0 + n)
                * (
                    8.0
                    + 8.0 * n
                    + 52.0 * np.power(n, 2)
                    + 93.0 * np.power(n, 3)
                    + 114.0 * np.power(n, 4)
                    + 13.0 * np.power(n, 5)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            + (48.0 * np.power(2.0 + n + np.power(n, 2), 2) * S21)
            / ((-1.0 + n) * n * (1.0 + n) * np.power(2.0 + n, 2))
            - (
                0.8888888888888888
                * (
                    432.0
                    - 452.0 * n
                    - 128.0 * np.power(n, 2)
                    + 977.0 * np.power(n, 3)
                    + 627.0 * np.power(n, 4)
                    + 195.0 * np.power(n, 5)
                    + 77.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                16.0
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * Sm2
            )
            / (n * np.power(1.0 + n, 3) * np.power(2.0 + n, 2))
            - (
                16.0
                * (
                    12.0
                    - 10.0 * n
                    - 54.0 * np.power(n, 2)
                    - 87.0 * np.power(n, 3)
                    - 40.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * Sm2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                64.0
                * (-2.0 - 14.0 * n - 13.0 * np.power(n, 2) + np.power(n, 3))
                * S1
                * Sm2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                16.0
                * (
                    8.0
                    + 20.0 * n
                    + 46.0 * np.power(n, 2)
                    + 27.0 * np.power(n, 3)
                    + 8.0 * np.power(n, 4)
                    + 3.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                16.0
                * (
                    -12.0
                    + 32.0 * n
                    + 39.0 * np.power(n, 2)
                    + 2.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * Sm21
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                (2.0 + n + np.power(n, 2))
                * (2.0 + 3.0 * n + 3.0 * np.power(n, 2))
                * (-7.212341418957566 - 8.0 * Sm3)
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                16.0
                * (
                    8.0
                    - 6.0 * n
                    - 5.0 * np.power(n, 2)
                    + 8.0 * np.power(n, 3)
                    + 3.0 * np.power(n, 4)
                )
                * Sm3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    28.849365675830263 * S1
                    - 4.0 * np.power(S1, 4)
                    - 36.0 * np.power(S1, 2) * S2
                    + 8.0 * np.power(S2, 2)
                    - 24.0 * S211
                    - 8.0 * S31
                    - 40.0 * np.power(S1, 2) * Sm2
                    - 12.0 * np.power(Sm2, 2)
                    + S1 * (48.0 * S21 - 32.0 * S3 + 16.0 * Sm21)
                    + 32.0 * Sm211
                    - 8.0 * Sm22
                    - 8.0 * S1 * Sm3
                    - 20.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 0.8888888888888888
        * (
            (57.69873135166053 * (-1.0 + n) * (-2.0 + 3.0 * n + 3.0 * np.power(n, 2)))
            / (np.power(n, 2) * np.power(1.0 + n, 2))
            - (
                0.5
                * (
                    96.0
                    + 400.0 * n
                    + 1296.0 * np.power(n, 2)
                    + 3168.0 * np.power(n, 3)
                    + 5062.0 * np.power(n, 4)
                    + 6277.0 * np.power(n, 5)
                    + 6853.0 * np.power(n, 6)
                    + 5026.0 * np.power(n, 7)
                    + 2368.0 * np.power(n, 8)
                    + 793.0 * np.power(n, 9)
                    + 149.0 * np.power(n, 10)
                )
            )
            / (np.power(n, 5) * np.power(1.0 + n, 5) * (2.0 + n))
            + (
                8.0
                * (
                    24.0
                    + 92.0 * n
                    + 172.0 * np.power(n, 2)
                    + 332.0 * np.power(n, 3)
                    + 529.0 * np.power(n, 4)
                    + 441.0 * np.power(n, 5)
                    + 179.0 * np.power(n, 6)
                    + 29.0 * np.power(n, 7)
                    + 2.0 * np.power(n, 8)
                )
                * S1
            )
            / (np.power(n, 4) * np.power(1.0 + n, 4) * (2.0 + n))
            - (
                8.0
                * (
                    22.0
                    + 79.0 * n
                    + 135.0 * np.power(n, 2)
                    + 160.0 * np.power(n, 3)
                    + 97.0 * np.power(n, 4)
                    + 26.0 * np.power(n, 5)
                    + 5.0 * np.power(n, 6)
                )
                * np.power(S1, 2)
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (64.0 * (1.0 + n) * np.power(S1, 3)) / (np.power(n, 2) * (2.0 + n))
            + (
                4.0
                * (
                    28.0
                    + 70.0 * n
                    + 96.0 * np.power(n, 2)
                    + 157.0 * np.power(n, 3)
                    + 75.0 * np.power(n, 4)
                    + 39.0 * np.power(n, 5)
                    + 23.0 * np.power(n, 6)
                )
                * S2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 3) * (2.0 + n))
            + (
                16.0
                * (
                    -4.0
                    - 18.0 * n
                    + 3.0 * np.power(n, 2)
                    + 10.0 * np.power(n, 3)
                    + np.power(n, 4)
                )
                * S1
                * S2
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (64.0 * (2.0 + n + np.power(n, 2)) * S21)
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            - (
                8.0
                * (2.0 + 5.0 * n + np.power(n, 2))
                * (2.0 - 1.0 * n + 3.0 * np.power(n, 2))
                * S3
            )
            / (np.power(n, 2) * np.power(1.0 + n, 2) * (2.0 + n))
            + (
                32.0
                * (
                    8.0
                    - 2.0 * n
                    + np.power(n, 2)
                    - 1.0 * np.power(n, 3)
                    + 4.0 * np.power(n, 4)
                    + 2.0 * np.power(n, 5)
                )
                * Sm2
            )
            / (np.power(n, 3) * np.power(1.0 + n, 2) * (2.0 + n))
            + ((2.0 - 1.0 * n + np.power(n, 2)) * (-128.0 * S1 * Sm2 + 128.0 * Sm21))
            / (np.power(n, 2) * (1.0 + n) * (2.0 + n))
            - (64.0 * (-1.0 + n) * Sm3) / (np.power(1.0 + n, 2) * (2.0 + n))
            - (
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    2.0 * np.power(S1, 4)
                    + 4.0 * np.power(S1, 2) * S2
                    + 6.0 * np.power(S2, 2)
                    + 24.0 * S211
                    + S1 * (-32.0 * S21 + 24.0 * S3)
                    - 8.0 * S31
                    + 20.0 * S4
                    + 16.0 * S2 * Sm2
                    + 8.0 * np.power(Sm2, 2)
                    - 16.0 * Sm22
                    + 32.0 * S1 * Sm3
                    - 32.0 * Sm31
                    + 40.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
        - 4.5
        * (
            (
                2.4041138063191885
                * (
                    -624.0
                    - 596.0 * n
                    - 956.0 * np.power(n, 2)
                    - 637.0 * np.power(n, 3)
                    - 111.0 * np.power(n, 4)
                    + 249.0 * np.power(n, 5)
                    + 83.0 * np.power(n, 6)
                )
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.14814814814814814
                * (
                    34560.0
                    + 211392.0 * n
                    + 604032.0 * np.power(n, 2)
                    + 1.099952e6 * np.power(n, 3)
                    + 1.506496e6 * np.power(n, 4)
                    + 1.640548e6 * np.power(n, 5)
                    + 1.596952e6 * np.power(n, 6)
                    + 1.332497e6 * np.power(n, 7)
                    + 702425.0 * np.power(n, 8)
                    - 16829.0 * np.power(n, 9)
                    - 322813.0 * np.power(n, 10)
                    - 155710.0 * np.power(n, 11)
                    + 66350.0 * np.power(n, 12)
                    + 101719.0 * np.power(n, 13)
                    + 47251.0 * np.power(n, 14)
                    + 10527.0 * np.power(n, 15)
                    + 939.0 * np.power(n, 16)
                )
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 5)
                * np.power(1.0 + n, 5)
                * np.power(2.0 + n, 5)
            )
            - (
                0.2962962962962963
                * (
                    432.0
                    + 7344.0 * n
                    + 4968.0 * np.power(n, 2)
                    - 26584.0 * np.power(n, 3)
                    - 33249.0 * np.power(n, 4)
                    - 44369.0 * np.power(n, 5)
                    - 71221.0 * np.power(n, 6)
                    - 71723.0 * np.power(n, 7)
                    - 31716.0 * np.power(n, 8)
                    + 13564.0 * np.power(n, 9)
                    + 22254.0 * np.power(n, 10)
                    + 9908.0 * np.power(n, 11)
                    + 2368.0 * np.power(n, 12)
                    + 296.0 * np.power(n, 13)
                )
                * S1
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
                    - 2016.0 * n
                    + 1132.0 * np.power(n, 2)
                    + 5904.0 * np.power(n, 3)
                    + 561.0 * np.power(n, 4)
                    - 4745.0 * np.power(n, 5)
                    - 448.0 * np.power(n, 6)
                    + 2710.0 * np.power(n, 7)
                    + 1897.0 * np.power(n, 8)
                    + 883.0 * np.power(n, 9)
                    + 170.0 * np.power(n, 10)
                )
                * np.power(S1, 2)
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
                    8.0
                    - 218.0 * n
                    - 139.0 * np.power(n, 2)
                    + 26.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * np.power(S1, 3)
            )
            / (n * np.power(1.0 + n, 2) * np.power(2.0 + n, 2))
            + (
                0.4444444444444444
                * (
                    -2016.0
                    - 6240.0 * n
                    - 5324.0 * np.power(n, 2)
                    - 2352.0 * np.power(n, 3)
                    - 861.0 * np.power(n, 4)
                    - 3047.0 * np.power(n, 5)
                    - 2692.0 * np.power(n, 6)
                    + 2506.0 * np.power(n, 7)
                    + 3091.0 * np.power(n, 8)
                    + 1213.0 * np.power(n, 9)
                    + 170.0 * np.power(n, 10)
                )
                * S2
            )
            / (
                np.power(-1.0 + n, 2)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                2.6666666666666665
                * (
                    -240.0
                    - 368.0 * n
                    - 254.0 * np.power(n, 2)
                    - 319.0 * np.power(n, 3)
                    - 285.0 * np.power(n, 4)
                    + 15.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * S1
                * S2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (2.0 + n + np.power(n, 2))
                * (
                    24.0
                    + 2.0 * n
                    + 13.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * S21
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                0.8888888888888888
                * (
                    -1188.0
                    - 1336.0 * n
                    - 1561.0 * np.power(n, 2)
                    - 899.0 * np.power(n, 3)
                    - 420.0 * np.power(n, 4)
                    + 165.0 * np.power(n, 5)
                    + 55.0 * np.power(n, 6)
                )
                * S3
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (-4.0 - 1.0 * n + np.power(n, 2))
                * (
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * Sm2
            )
            / ((-1.0 + n) * n * np.power(1.0 + n, 3) * np.power(2.0 + n, 3))
            + (
                1.7777777777777777
                * (
                    648.0
                    + 2916.0 * n
                    + 3802.0 * np.power(n, 2)
                    + 2731.0 * np.power(n, 3)
                    + 1381.0 * np.power(n, 4)
                    + 1517.0 * np.power(n, 5)
                    + 2086.0 * np.power(n, 6)
                    + 1508.0 * np.power(n, 7)
                    + 597.0 * np.power(n, 8)
                    + 94.0 * np.power(n, 9)
                )
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 3)
                * np.power(1.0 + n, 3)
                * np.power(2.0 + n, 3)
            )
            - (
                32.0
                * (
                    -12.0
                    + 4.0 * n
                    + 13.0 * np.power(n, 2)
                    - 32.0 * np.power(n, 3)
                    - 23.0 * np.power(n, 4)
                    + 2.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (
                    -48.0
                    - 68.0 * n
                    - 24.0 * np.power(n, 2)
                    - 49.0 * np.power(n, 3)
                    + 34.0 * np.power(n, 4)
                    + 11.0 * np.power(n, 5)
                )
                * S1
                * Sm2
            )
            / ((-1.0 + n) * np.power(n, 2) * (1.0 + n) * np.power(2.0 + n, 2))
            + (
                5.333333333333333
                * (
                    -72.0
                    - 92.0 * n
                    - 194.0 * np.power(n, 2)
                    - 361.0 * np.power(n, 3)
                    - 189.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
                )
                * Sm21
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
                    -24.0
                    - 46.0 * n
                    - 35.0 * np.power(n, 2)
                    + 22.0 * np.power(n, 3)
                    + 11.0 * np.power(n, 4)
                )
                * (-2.4041138063191885 - 2.6666666666666665 * Sm3)
            )
            / (
                (-1.0 + n)
                * np.power(n, 2)
                * np.power(1.0 + n, 2)
                * np.power(2.0 + n, 2)
            )
            - (
                5.333333333333333
                * (
                    -108.0
                    - 176.0 * n
                    - 263.0 * np.power(n, 2)
                    - 247.0 * np.power(n, 3)
                    - 114.0 * np.power(n, 4)
                    + 33.0 * np.power(n, 5)
                    + 11.0 * np.power(n, 6)
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
                4.0
                * (2.0 + n + np.power(n, 2))
                * (
                    -28.849365675830263 * S1
                    + 2.0 * np.power(S1, 4)
                    + 32.0 * np.power(S1, 2) * S2
                    + 2.0 * np.power(S2, 2)
                    + 16.0 * S31
                    + 4.0 * S4
                    + (40.0 * np.power(S1, 2) + 16.0 * S2) * Sm2
                    + 12.0 * np.power(Sm2, 2)
                    + S1 * (-16.0 * S21 + 40.0 * S3 - 80.0 * Sm21)
                    + 96.0 * Sm211
                    - 56.0 * Sm22
                    + 72.0 * S1 * Sm3
                    - 64.0 * Sm31
                    + 44.0 * Sm4
                )
            )
            / (n * (1.0 + n) * (2.0 + n))
        )
    )
    return a_Hg_l0 + a_Hg_l1 * L + a_Hg_l2 * L**2 + a_Hg_l3 * L**3
