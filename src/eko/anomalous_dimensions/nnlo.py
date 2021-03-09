# -*- coding: utf-8 -*-
"""
This file contains the next-to-next-to-leading-order Altarelli-Parisi splitting kernels.

These expression have been obtained from:
https://www.liverpool.ac.uk/~avogt/p2mom.f
:cite:`Moch:2004pa`.
"""
import numba as nb

from . import harmonics
#from .. import constants


# TODO: is sign convention correct? 

# abbreviation for NS parts
def E1(n):
    S1 = harmonics.harmonic_S1(n)
    S2 = harmonics.harmonic_S2(n)
    zeta2 = harmonics.zeta2

    return S1 / n ** 2 + (S2 - zeta2) / n

# abbreviation for NS parts
def E2(n):
    S1 = harmonics.harmonic_S1(n)
    S2 = harmonics.harmonic_S2(n)
    S3 = harmonics.harmonic_S3(n)
    zeta2 = harmonics.zeta2
    zeta3 = harmonics.zeta3

    return 2.0 * ( - S1 / n ** 3 + (zeta2 - S2) / n ** 2 - (S3 - zeta3) / n )

# common part for NS plus and NS minus
@nb.njit("c16(c16,u1)", cache=True)
def pf2_nfnf(n):

    S1 = harmonics.harmonic_S1(n)
    S2 = harmonics.harmonic_S2(n)
    S3 = harmonics.harmonic_S3(n)

    res = - ( 17.0 / 72.0 - 2.0 / 27.0 * S1 - 10.0 / 27.0 * S2 \
        + 2.0 / 9.0 * S3 - ( 12.0 * n ** 4 + 2.0 * n ** 3 - 12.0 * n ** 2 \
            - 2.0 * n + 3.0 ) / ( 27.0 * n ** 3 * (n + 1.0) ** 3) ) * 32.0 / 3.0
    return res


@nb.njit("c16(c16,u1)", cache=True)
def gamma_nsm_2(n, nf: int):
    """
    Computes the next-to-next-to-leading-order valence-like non-singlet anomalous dimension.

    Implements Eq. (3.8) of :cite:`Moch:2004pa`.

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_nsm_2 : complex
            Next-to-next-to-leading-order valence-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,-}^{(2)}(N)`
    """
    S1 = harmonics.harmonic_S1(n)

    # The QCD colour factors have been hard-wired in the parametrizations.

    pm2 = - 1174.898 * ( S1 + 1.0 / n ) + 1295.470 - 714.1 * S1 / n - 433.2 / (n + 3.0) \
        + 297.0 / (n + 2.0) - 3505.0 / (n + 1.0) + 1860.2 / n - 1465.2 / n ** 2 \
            + 399.2 * 2.0 / n ** 3 - 320.0 / 9.0 * 6.0 / n ** 4 - 116.0 / 81.0 * 6.0 / (n + 1.0) ** 4 \
                + 684.0 * E1(n) + 251.2 * E2(n)

    pm2_nf = + 183.187 * ( S1 + 1.0 / n ) - 173.933 + 5120 / 81.0 * S1 / n \
        + 34.76 / (n + 3.0) + 77.89 / (n + 2.0) + 406.5 / (n + 1.0) - 216.62 / n \
            + 172.69 / n ** 2 - 3216.0 / 81.0 * 2.0 / n ** 3 + 256.0 / 81.0 * 6.0 / n ** 4 \
                - 65.43 * E1(n) + 1.136 * 6.0 / (n + 1.0) ** 4

    result = pm2 + nf * pm2_nf + nf ** 2 * pf2_nfnf(n)
    return result

@nb.njit("c16(c16,u1)", cache=True)
def gamma_nsp_2(n, nf: int):
    """
    Computes the next-to-next-to-leading-order singlet-like non-singlet anomalous dimension.

    Implements Eq. (3.7) of :cite:`Moch:2004pa`.

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_nsp_2 : complex
            Next-to-next-to-leading-order singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,+}^{(2)}(N)`
    """
    S1 = harmonics.harmonic_S1(n)

    # The QCD colour factors have been hard-wired in the parametrizations.

    pp2 = - 1174.898 * ( S1 + 1.0 / n ) + 1295.384 - 714.1 * S1 / n - 522.1 / (n + 3.0) \
        + 243.6 / (n + 2.0) - 3135.0/ (n + 1.0) + 1641.1 / n - 1258.0 / n ** 2 \
            + 294.9 * 2.0 / n ** 3 - 800 / 27.0 * 6.0 / n ** 4 + 128 / 81.0 * 24.0 / n ** 5 \
                + 563.9 * E1(n) + 256.8 * E2(n)

    pp2_nf = + 183.187 * ( S1 + 1.0 / n ) - 173.924 + 5120 / 81.0 * S1 / n \
        + 44.79 / (n + 3.0) + 72.94 / (n + 2.0) + 381.1 / (n + 1.0) - 197.0 / n \
            + 152.6 / n ** 2 - 2608.0 / 81.0 * 2.0 / n ** 3 + 192.0 / 81.0 * 6.0 / n ** 4 \
                - 56.66 * E1(n) + 1.497 * 6.0 / (n + 1.0) ** 4

    result = pp2 + nf * pp2_nf + nf ** 2 * pf2_nfnf(n)
    return result


@nb.njit("c16(c16,u1)", cache=True)
def gamma_nsv_2(n, nf: int):
    """
    Computes the next-to-next-to-leading-order singlet-like non-singlet anomalous dimension.

    Implements Eq. (3.9) of :cite:`Moch:2004pa`.

    Parameters
    ----------
        n : complex
            Mellin moment
        nf : int
            Number of active flavors

    Returns
    -------
        gamma_nsv_2 : complex
            Next-to-next-to-leading-order singlet-like non-singlet anomalous dimension
            :math:`\\gamma_{ns,v}^{(2)}(N)`
    """
    S1 = harmonics.harmonic_S1(n)
    zeta2 = harmonics.zeta2


    B11 = - ( S1 + 1.0 / (n + 1.0) ) / (n + 1.0)
    B12 = - ( S1 + 1.0 / (n + 1.0) + 1.0 / (n + 2.0) ) / (n + 2.0)

    # TODO: is this needed ? 
    # with special care for the first moment of x^-1 ln(1-x)
    if abs( n.imag ) < 1.0e-5 and abs( n - 1.0 ) < 1.0e-5 :
        B1M = - zeta2
    else:
        B1M = - ( S1 - 1.0 / n ) / (n - 1.0)

    ps2 = - 163.9 * (B1M + S1 / n) - 7.208 * (B11 - B12) + 4.82 * ( 1.0 / (n + 3.0) - 1.0 / (n + 4.0) ) \
        - 43.12 * ( 1.0 / (n + 2.0) - 1.0 / (n + 3.0) ) + 44.51 * ( 1.0 / (n + 1.0) - 1.0 / (n + 2.0) ) + 151.49 * ( 1.0 / n - 1.0 / (n + 1.0) ) \
            - 178.04 / n ** 2 + 6.892 * 2.0 / n ** 3 - 40.0 / 27.0 * ( - 2.0 * 6.0 / n ** 4 - 24.0 / n ** 5 ) \
                - 173.1 * E1(n) + 46.18 * E2(n)

    result = gamma_nsm_2(n, nf) + nf * ps2
    return result


#######################
# temp dict TO PYTHON
#######################
    # NI = 1.0 / n
    # NI2 = 1.0 / n ** 2
    # NI3 = 1.0 / n ** 3 
    # NM = n - 1.0
    # NMI = 1.0 / (n - 1.0)
    # NMI2 = 1.0 / (n - 1.0) ** 2
    # N1 = n + 1.0
    # N1I = 1.0 / (n + 1.0)
    # N1I2 = 1.0 / (n + 1.0) ** 2 
    # N1I3 = 1.0 / (n + 1.0) ** 3 
    # N2 = n + 2.0
    # N2I = 1.0 / (n + 2.0)
    # S1M = S1 - 1.0 / n
    # S11 = S1 + 1.0 / (n + 1.0)
    # S12 = S1 + 1.0 / (n + 1.0) + 1.0 / (n + 2.0)

    # A0 = - ( S1 + 1.0 / n )
    # B1 = - S1 / n
    # B11 = - ( S1 + 1.0 / (n + 1.0) ) / (n + 1.0)
    # B12 = - ( S1 + 1.0 / (n + 1.0) + 1.0 / (n + 2.0) ) / (n + 2.0)

    # C0 = 1.0 / n
    # C1 = 1.0 / (n + 1.0)
    # C2 = 1.0 / (n + 2.0)
    # C3 = 1.0 / (n + 3.0)
    # C4 = 1.0 / (n + 4.0)

    # D1 = - 1.0 / n ** 2 
    # D2 = 2.0 / n ** 3 
    # D3 = - 6.0 / n ** 4 
    # D31 = - 6.0 / (n + 1.0) ** 4
    # D4 = 24.0 / n ** 5
#######################

# * =====================================================================
# *
# *    
#        SUBROUTINE P2NSP ( P2PLSN, P2MINN, P2VALN, N, NF )
# *
#        IMPLICIT DOUBLE COMPLEX (A - Z)
#        DOUBLE PRECISION EMC, ZETA2, ZETA3, ZETA4
#        INTEGER NF
#        PARAMETER ( EMC  = 0.57721 56649D0, ZETA2 = 1.64493 40668D0,
#      ,            ZETA3 = 1.20205 69032D0, ZETA4 = 1.08232 32337D0 )
# *
# * ..Analytic continuations of simple harmonic sums
# *
#        HS1(Z) =  EMC  + PSI (Z+1.) 
#        HS2(Z) = ZETA2 - DPSI (Z+1.,1)
#        HS3(Z) = ZETA3 + 0.5 * DPSI (Z+1.,2)
#        HS4(Z) = ZETA4 - 1./6.D0 * DPSI (Z+1.,3)
# *
# * ---------------------------------------------------------------------
# *
# * ..Some abbreviations
# *
#        NI = 1./N
#        NI2 = NI*NI
#        NI3 = NI*NI2
#        NM = N - 1.
#        NMI = 1./NM
#        NMI2 = NMI*NMI
# *
#        N1 = N + 1.
#        N1I = 1./N1
#        N1I2 = N1I*N1I
#        N1I3 = N1I*N1I2
#        N2 = N + 2.
#        N2I = 1./N2
# *
#        S1 = HS1(N)
#        S2 = HS2(N)
#        S3 = HS3(N)
#        S1M = S1 - NI
#        S11 = S1 + N1I
#        S12 = S11 + N2I
# *
# * ---------------------------------------------------------------------
# *
# * ..The moments of the functions employed in the parametrizations:
# *
# * ...1/(1-x)_+ [A0]  and  x^a ln^b (1-x) [B`b(a)' with M for a = -1]
# *
#        A0  = - S1M
#        B1  = - S1 * NI
# * ...with special care for the first moment of x^-1 ln(1-x)
#        IF ( ( DABS (DIMAG(N)) .LT. 1.D-5 ) .AND. 
#      ,      ( DABS ( DBLE(N) - 1.D0 ) .LT. 1.D-5 ) ) THEN
#          B1M = - ZETA2
#        ELSE
#          B1M = - S1M * NMI
#        ENDIF
#        B11 = - S11 * N1I
#        B12 = - S12 * N2I
# *
# * ...x^a [C`a'] 
# *
#        C0 = NI
#        C1 = N1I
#        C2 = N2I
#        C3 = 1./(N+3.)
#        C4 = 1./(N+4.)
# *
# * ...x^a ln^b x [D`b(a)'] 
# *
#        D1  = - NI2
#        D2  = 2.* NI3
#        D3  = - 6.* NI2*NI2
#        D31 = - 6.* N1I2*N1I2
#        D4  = 24.* NI2*NI3
# *
# * ...x^a ln^b x ln(1-x) [E`b(a)']
# *
#        E1  = S1*NI2 + (S2-ZETA2)*NI
#        E2  = 2.* ( - S1*NI3 + (ZETA2-S2)*NI2 - (S3-ZETA3)*NI )
# *
# * ---------------------------------------------------------------------
# *
# * ..The parametrized n_f^{0,1} components of P_ns^(2)i, i = +, -, v. 
# *
#        PP2 = + 1174.898 * A0 + 1295.384 + 714.1 * B1 - 522.1 * C3
#      1       + 243.6 * C2 - 3135.* C1 + 1641.1 * C0 + 1258.* D1
#      2       + 294.9 * D2 + 800/27.D0 * D3 + 128/81.D0 * D4
#      3       + 563.9 * E1 + 256.8 * E2
#      4     + NF * ( - 183.187 * A0 - 173.924 - 5120/81.D0 * B1
#      5       + 44.79 * C3 + 72.94 * C2 + 381.1 * C1 - 197.0 * C0
#      6       - 152.6 * D1 - 2608./81.D0 * D2 - 192./81.D0 * D3
#      7       - 56.66 * E1 - 1.497 * D31 )
# *
#        PM2 = + 1174.898 * A0 + 1295.470 + 714.1 * B1 - 433.2 * C3
#      1       + 297.0 * C2 - 3505.*C1 + 1860.2 * C0 + 1465.2 * D1
#      2       + 399.2 * D2 + 320./9.D0 * D3 + 116./81.D0 * D4
#      3       + 684.0 * E1 + 251.2 * E2
#      4     + NF * ( - 183.187 * A0 - 173.933 - 5120/81.D0 * B1
#      5       + 34.76 * C3 + 77.89 * C2 + 406.5 * C1 - 216.62 * C0
#      6       - 172.69 * D1 - 3216./81.D0 * D2 - 256./81.D0 * D3
#      7       - 65.43 * E1 - 1.136 * D31 )
# *
#        PS2 = - 163.9 * (B1M-B1)-7.208 * (B11-B12) + 4.82 * (C3-C4)
#      1       - 43.12 * (C2-C3) + 44.51 * (C1-C2) + 151.49 * (C0-C1)
#      2       + 178.04 * D1 + 6.892 * D2 - 40./27.D0 * (2.*D3 - D4)
#      2       - 173.1 * E1 + 46.18 * E2
# *
# * ..The exact n_f^2 contribution first determined by J.A. Gracey in
# *    Phys. Lett. B322 (1994) 141
# *
#        PF2 = - ( 17./72.D0 - 2./27.D0 * S1 - 10./27.D0 * S2
#      1        + 2./9.D0 * S3 - (12.* N**4 + 2.* N**3 - 12.* N**2
#      2        - 2.* N + 3.)/(27.* N**3 * N1**3) ) * 32./3.D0
# *
# * ---------------------------------------------------------------------
# *
# * ..Assemble the pieces for the output
# *
#        P2PLSN = PP2 + NF**2 * PF2
#        P2MINN = PM2 + NF**2 * PF2
#        P2VALN = P2MINN + NF * PS2
# *
#        RETURN
#        END
# *
# * =====================================================================
# *























# * =====================================================================
# *
# *    
#        SUBROUTINE P2IJP ( P2PSN, P2QGN, P2GQN, P2GGN, N, NF )
# *
#        IMPLICIT DOUBLE COMPLEX (A - Z)
#        DOUBLE PRECISION EMC, ZETA2, ZETA3, ZETA4
#        INTEGER NF
#        PARAMETER ( EMC  = 0.57721 56649D0, ZETA2 = 1.64493 40668D0,
#      ,            ZETA3 = 1.20205 69032D0, ZETA4 = 1.08232 32337D0 )
# *
# * ..Analytic continuations of simple harmonic sums
# *
#        HS1(Z) =  EMC  + PSI (Z+1.) 
#        HS2(Z) = ZETA2 - DPSI (Z+1.,1)
#        HS3(Z) = ZETA3 + 0.5 * DPSI (Z+1.,2)
#        HS4(Z) = ZETA4 - 1./6.D0 * DPSI (Z+1.,3)
# *
# * ---------------------------------------------------------------------
# *
# * ..Some abbreviations
# *
#        NI = 1./N
#        NI2 = NI*NI
#        NI3 = NI*NI2
#        NM = N - 1.
#        NMI = 1./NM
#        NMI2 = NMI*NMI
# *
#        N1 = N + 1.
#        N1I = 1./N1
#        N1I2 = N1I*N1I
#        N1I3 = N1I*N1I2
#        N2 = N + 2.
#        N2I = 1./N2
# *
#        S1 = HS1(N)
#        S2 = HS2(N)
#        S3 = HS3(N)
#        S4 = HS4(N)
#        S1M = S1 - NI
#        S11 = S1 + N1I
#        S2M = S2 - NI2
#        S21 = S2 + N1I2
#        S31 = S3 + N1I3
# *
# * ---------------------------------------------------------------------
# *
# * ..The moments of the functions employed in the parametrizations:
# *
# * ...1/(1-x)_+ [A0]  and  x^a ln^b (1-x) [B`b(a)' with M for a = -1]
# *
#        A0  = - S1M
#        B1  = - S1 * NI
#        B1M = - S1M * NMI
#        B11 = - S11 * N1I
#        B2  = (S1**2 + S2) * NI
#        B2M = (S1M**2 + S2M) * NMI
#        B21 = (S11**2 + S21) * N1I
#        B3  = - (S1**3 + 3.*S1*S2 + 2.*S3) * NI
#        B31 = - (S11**3 + 3.*S11*S21 + 2.*S31) * N1I
#        B4  = (S1**4 + 6.*S1**2*S2 + 8.*S1*S3 + 3.*S2**2 + 6.*S4) * NI
# *
# * ...x^a [C`a'] 
# *
#        C0 = NI
#        CM = NMI
#        C1 = N1I
#        C2 = N2I
#        C3 = 1./(N+3.)
#        C4 = 1./(N+4.)
# *
# * ...x^a ln^b x [D`b(a)'] 
# *
#        D1  = - NI2
#        D1M = - NMI2
#        D11 = - N1I2
#        D2  = 2.* NI3
#        D21 = 2.* N1I3
#        D3  = - 6.* NI2*NI2
#        D31 = - 6.* N1I2*N1I2
#        D4  = 24.* NI2*NI3
#        D41 = 24.* N1I2*N1I3
# *
# * ...x^a ln^b x ln(1-x) [E`b(a)']
# *
#        E1  = S1*NI2 + (S2-ZETA2)*NI
#        E11 = S11*N1I2 + (S21-ZETA2)*N1I
#        E2  = 2.* ( - S1*NI3 + (ZETA2-S2)*NI2 - (S3-ZETA3)*NI )
# *
# * ---------------------------------------------------------------------
# *
# * ..The parametrized n_f-components of P_ps^(2) and P_qg^(2)
# *   [ P_qq^(2) is obtained by adding the non-singlet quantity 
# *     P_ns^(2) provided by the subroutine  P2NSP  to P_ps^(2) ]
# *
#        PS1 = - 3584./27.* (D1M-D1) - 506.* (CM-C0) + 160./27.* (D4-D41)
#      ,       - 400./9.* (D3-D31) + 131.4 * (D2-D21) - 661.6 * (D1-D11)
#      ,       - 5.926 * (B3-B31) - 9.751 * (B2-B21) - 72.11 * (B1-B11)
#      ,       + 177.4 * (C0-C1) + 392.9 * (C1-C2) - 101.4 * (C2-C3)
#      ,       - 57.04 * (E1-E11)
#        PS2 = 256./81.* (CM-C0) + 32./27.* (D3-D31) + 17.89 * (D2-D21)
#      ,       + 61.75 * (D1-D11) + 1.778 * (B2-B21) + 5.944 * (B1-B11)
#      ,       + 100.1 * (C0-C1) - 125.2 * (C1-C2) + 49.26 * (C2-C3)
#      ,       - 12.59 * (C3-C4) - 1.889 * (E1-E11)
# *
#        QG1 = - 896./3.* D1M - 1268.3 * CM + 536./27.* D4 - 44./3.* D3 
#      ,       + 881.5 * D2 + 424.9 * D1 + 100./27.* B4 - 70./9.* B3 
#      ,       - 120.5 * B2 + 104.42 * B1 + 2522.* C0 - 3316. * C1 
#      ,       + 2126. * C2 + 1823. * E1 - 25.22 * E2 - 252.5 * D31
#        QG2 =   1112./243.* CM - 16./9.* D4 - 376./27.* D3 - 90.8 * D2
#      ,       - 254.0 * D1 + 20./27.* B3 + 200./27.* B2 - 5.496 * B1
#      ,       - 252.0 * C0 + 158.0 * C1 + 145.4 * C2 - 139.28 * C3
#      ,       - 53.09 * E1 - 80.616 * E2 - 98.07 * D21 + 11.70 * D31
# *
# * ...and of P^(2)_gq and P^(2)_gg  [GQ2 is exact]
# *
#        GQ0 = 1189.3 * D1M + 6163.1 * CM - 4288./81. * D4 + 1568./9.* D3
#      ,       - 1794.* D2 + 4033.* D1 + 400./81.* B4 + 2200./27.* B3
#      ,       + 606.3 * B2 + 2193.* B1 - 4307.* C0 + 489.3 * C1 
#      ,       + 1452.* C2 + 146.* C3 - 447.3 * E2 - 972.9 * D21 
#        GQ1 =   71.082 * D1M - 46.41 * CM + 128./27.* D4 + 704/81.* D3
#      ,       + 20.39 * D2 + 174.8 * D1 - 400./81.* B3 - 68.069 * B2 
#      ,       - 296.7 * B1 - 183.8 * C0 + 33.35 * C1 - 277.9 * C2 
#      ,       + 108.6 * D21 - 49.68 * E1
#        GQ2 = ( 64.* (- CM + C0 + 2.* C1) + 320.* (B1M - B1 + 0.8 * B11)
#      ,       + 96.* (B2M - B2 + 0.5 * B21) ) / 27. 
# *
#        GG0 = 2675.8 * D1M + 14214.* CM - 144.D0 * D4 + 72.D0 * D3
#      ,       - 7471.* D2 + 274.4 * D1 - 20852.* C0 + 3968.* C1 
#      ,       - 3363.* C2 + 4848.* C3 + 7305.* E1 + 8757.* E2 
#      ,       + 3589.* B1 + 4425.894 + 2643.521 * A0 
#        GG1 =   157.27 * D1M + 182.96 * CM + 512./27.D0 * D4 
#      ,       + 832./9.D0 * D3 + 491.3 * D2 + 1541.* D1 - 350.2 * C0
#      ,       + 755.7 * C1 - 713.8 * C2 + 559.3 * C3 + 26.15 * E1
#      ,       - 808.7 * E2 - 320.D0 * B1 - 528.723 - 412.172 * A0
#        GG2 = - 680./243.D0 * CM - 32./27.D0 * D3 + 9.680 * D2 
#      ,       - 3.422 * D1 - 13.878 * C0 + 153.4 * C1 - 187.7 * C2
#      ,       + 52.75 * C3 - 115.6 * E1 + 85.25 * E11 - 63.23 * E2
#      ,       + 6.4630 - 16./9.D0 * A0
# *
# * ---------------------------------------------------------------------
# *
# * ..Assemble the pieces for the output
# *
#        P2PSN =       NF * ( PS1 + NF * PS2 )
#        P2QGN =       NF * ( QG1 + NF * QG2 )
#        P2GQN = GQ0 + NF * ( GQ1 + NF * GQ2 )
#        P2GGN = GG0 + NF * ( GG1 + NF * GG2 )
# *
#        RETURN
#        END
# *
# * =================================================================av==