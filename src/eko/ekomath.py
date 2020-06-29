# -*- coding: utf-8 -*-
"""
    Implements higher mathematical functions.

    The functions are need in :doc:`Mellin space </Theory/Mellin>`.
"""

import numpy as np
import numba as nb
from eko import t_complex


@nb.njit
def cern_polygamma(
    Z: t_complex, K: int
):  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """
      Computes the polygamma functions :math:`\\psi_k(z)`.

      Reimplementation of ``WPSIPG`` (C317) in `CERNlib <http://cernlib.web.cern.ch/cernlib/>`_
      :cite:`KOLBIG1972221`.

      Note that the `SciPy implementation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html>`_
      does not allow for complex inputs.

      Parameters
      ----------
        Z : t_complex
          argument of polygamma function
        K : int
          order of polygamma function

      Returns
      -------
        H : t_complex
          k-th polygamma function :math:`\\psi_k(z)`
    """
    # fmt: off
    DELTA = 5e-13
    R1 = 1
    HF = R1/2
    C1 = np.pi**2
    C2 = 2*np.pi**3
    C3 = 2*np.pi**4
    C4 = 8*np.pi**5

    # SGN is originally indexed 0:4 -> no shift
    SGN = [-1,+1,-1,+1,-1]
    # FCT is originally indexed -1:4 -> shift +1
    FCT = [0,1,1,2,6,24]

    # C is originally indexed 1:6 x 0:4 -> swap indices and shift new last -1
    C = nb.typed.List()
    C.append([
            8.33333333333333333e-2,
           -8.33333333333333333e-3,
            3.96825396825396825e-3,
           -4.16666666666666667e-3,
            7.57575757575757576e-3,
           -2.10927960927960928e-2])
    C.append([
            1.66666666666666667e-1,
           -3.33333333333333333e-2,
            2.38095238095238095e-2,
           -3.33333333333333333e-2,
            7.57575757575757576e-2,
           -2.53113553113553114e-1])
    C.append([
            5.00000000000000000e-1,
           -1.66666666666666667e-1,
            1.66666666666666667e-1,
           -3.00000000000000000e-1,
            8.33333333333333333e-1,
           -3.29047619047619048e+0])
    C.append([
            2.00000000000000000e+0,
           -1.00000000000000000e+0,
            1.33333333333333333e+0,
           -3.00000000000000000e+0,
            1.00000000000000000e+1,
           -4.60666666666666667e+1])
    C.append([10., -7., 12., -33., 130., -691.])
    U=Z
    X=np.real(U)
    A=np.abs(X)
    if K < 0 or K > 4:
        raise NotImplementedError("Order K has to be in [0:4]")
    if np.abs(np.imag(U)) < DELTA and np.abs(X+int(A)) < DELTA:
        raise ValueError("Argument Z equals non-positive integer")
    K1=K+1
    if X < 0:
        U=-U
    V=U
    H=0
    if A < 15:
        H=1/V**K1
        for I in range(1,14-int(A)+1):
            V=V+1
            H=H+1/V**K1
        V=V+1
    R=1/V**2
    P=R*C[K][6-1]
    for I in range(5,1-1,-1):
        P=R*(C[K][I-1]+P)
    H=SGN[K]*(FCT[K+1]*H+(V*(FCT[K-1+1]+P)+HF*FCT[K+1])/V**K1)
    if K == 0:
        H=H+np.log(V)
    if X < 0:
        V=np.pi*U
        X=np.real(V)
        Y=np.imag(V)
        A=np.sin(X)
        B=np.cos(X)
        T=np.tanh(Y)
        P=np.complex(B,-A*T)/np.complex(A,B*T)
        if K == 0:
            H=H+1/U+np.pi*P
        elif K == 1:
            H=-H+1/U**2+C1*(P**2+1)
        elif K == 2:
            H=H+2/U**3+C2*P*(P**2+1)
        elif K == 3:
            R=P**2
            H=-H+6/U**4+C3*((3*R+4)*R+1)
        elif K == 4:
            R=P**2
            H=H+24/U**5+C4*P*((3*R+5)*R+2)
    return H
    # fmt: on


@nb.njit
def harmonic_S1(N: t_complex):
    r"""
      Computes the simple harmonic sum.

      .. math::
        S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E

      with :math:`\psi_0(N)` the digamma function and :math:`\gamma_E` the
      Euler-Mascheroni constant.

      Parameters
      ----------
        N : t_complex
          Mellin moment

      Returns
      -------
        S_1 : t_complex
          (simple) Harmonic sum :math:`S_1(N)`

      See Also
      --------
        cern_polygamma :
    """
    return cern_polygamma(N + 1.0, 0) + np.euler_gamma
