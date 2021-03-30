# *
# * ..File: ans2mom.f     
# *
# *
# * ..The subroutine  ANS2MOM  for the NNLO (alpha_s^2) heavy quark 
# *    contribution  A2NS  to the non-singlet operator matrix element 
# *    (OME) in N-space in the MS(bar) scheme for mu_f^2 = m_H^2.
# *    The coupling constant is normalized as  a_s = alpha_s/(4*pi).
# *
# * ..This quantity, presented in Appendix B of Buza, Matiounine, Smith 
# *    and van Neerven, Eur. Phys. J. C1 (1998) 301 (BSMN), is required 
# *    for the N_f matching of the NNLO parton densities.
# *
# * ..The results (written to the common-block  ANS2)  are computed on an 
# *    external NDIM-dimensional array  NA  of complex Mellin moments 
# *    provided by the common-block  MOMS. 
# *
# * ..The SU(N_colours=3) colour factors  CF, CA and TF  are taken from
# *    the common-block  COLOUR.  The simple harmonic sums S_i(N) are
# *    provided by the common-block  HSUMS,  and the lowest integer
# *    values of the Riemann Zeta-function are provided by  RZETA.
# *
# * =====================================================================
# *
# *
#        SUBROUTINE ANS2MOM 
# *
#        IMPLICIT DOUBLE COMPLEX (A - Z)
#        INTEGER NMAX, NDIM, KN
#        PARAMETER (NDIM = 144)
#        DOUBLE PRECISION ZETA(6), CF, CA, TR
# *
# * ---------------------------------------------------------------------
# *
# * ..Input common-blocks 
# *
#        COMMON / MOMS   / NA (NDIM)
#        COMMON / NNUSED / NMAX
#        COMMON / HSUMS  / S(NDIM,6)
#        COMMON / RZETA  / ZETA
#        COMMON / COLOUR / CF, CA, TR
# *
# * ..Output common-block 
# *
#        COMMON / ANS2   / A2NS (NDIM)
# *
# * ---------------------------------------------------------------------
# *
# * ..Begin of the Mellin-N loop
# *
#        DO 1 KN = 1, NMAX
# *
# * ..Some abbreviations
# *
#        N  = NA(KN)
#        S1 = S(KN,1)
#        S2 = S(KN,2)
#        S3 = S(KN,3)
# *
#        N1 = N + 1.
#        NI = 1./N
#        N1I = 1./N1
# *
#        S1M = S1 - NI
#        S2M = S2 - NI*NI
#        S3M = S3 - NI**3
#        S21 = S2 + N1I*N1I
#        S31 = S3 + N1I**3
# *
# * ---------------------------------------------------------------------
# *
# *  ..Moments of the basic x-space functions 
# *
#        A0 = - S1M
# *
#        C0 = NI
#        C1 = N1I
# *
#        D1  = - NI*NI
#        D11 = - N1I*N1I
# *
#        G1  = S2M - ZETA(2)  
#        G12 = S21 - ZETA(2)  
#        G2  = - 2.* ( S3M - ZETA(3) )  
#        G22 = - 2.* ( S31 - ZETA(3) )  
# *
# * ---------------------------------------------------------------------
# *
# * ..The moments of the OME A_{qq,H}^{NS,(2)} given in Eq. (B.4) of BMSN 
# *
#        A2QQ = 224./27.D0 * A0 - 8./3.D0 * ZETA(3) + 40/9.D0 * ZETA(2) 
#      1        + 73./18.D0 + 44./27.D0 * C0 - 268./27.D0 * C1 
#      2        + 8./3.D0 * (D1 - D11) + 20./9.D0 * (G1 + G12) 
#      3        + 2./3.D0 * (G2 + G22)
# *
# * ..Output to the array 
# *
#        A2NS(KN) = CF*TR * A2QQ
# *
# * ---------------------------------------------------------------------
# *
#   1    CONTINUE
# *
#        RETURN
#        END
# *
# * =================================================================av==


# *
# * ..File: asg2mom.f   
# *
# *
# * ..The subroutine  ASG2MOM  for the NNLO (alpha_s^2) heavy-quark 
# *    singlet operator matrix elements (OME's)  A2SG  in N-space in the 
# *    MS(bar) scheme for mu_f^2 = m_H^2.
# *    The coupling constant is normalized as  a_s = alpha_s/(4*pi).
# *
# * ..These quantities, presented in Appendix B of Buza, Matiounine, 
# *    Smith and van Neerven, Eur. Phys. J. C1 (1998) 301 (BSMN), are 
# *    required for the N_f matching of the NNLO parton densities.
# *
# * ..The results (written to the common-block  ASG2)  are computed on an 
# *    external NDIM-dimensional array  NA  of complex Mellin moments 
# *    provided by the common-block  MOMS.  The notation for the last two
# *    array arguments is  1 = H  (1 = q)  when `1' is the second (third)
# *    argument and  2 = g.  
# *
# * ..The SU(N_colours=3) colour factors  CF, CA and TF  are taken from
# *    the common-block  COLOUR.  The simple harmonic sums S_i(N) are
# *    provided by the common-block  HSUMS,  and the lowest integer 
# *    values of the Riemann Zeta-function are provided by  RZETA.
# *    A2SG(KN,1,2) is calculated via a compact parametrization in which
# *    the colour factors are hard-wired.
# *
# * =====================================================================
# *
# *
#        SUBROUTINE ASG2MOM 
# *
#        IMPLICIT DOUBLE COMPLEX (A - Z)
#        INTEGER NMAX, NDIM, KN
#        PARAMETER (NDIM = 144)
#        DOUBLE PRECISION ZETA(6), CF, CA, TR
# *
# * ---------------------------------------------------------------------
# *
# * ..Input common-blocks 
# *
#        COMMON / MOMS   / NA (NDIM)
#        COMMON / NNUSED / NMAX
#        COMMON / HSUMS  / S(NDIM,6)
#        COMMON / RZETA  / ZETA
#        COMMON / COLOUR / CF, CA, TR
# *
# * ..Output common-block 
# *
#        COMMON / ASG2   / A2SG (NDIM, 2, 2)
# *
# * ---------------------------------------------------------------------
# *
# * ..Begin of the Mellin-N loop
# *
#        DO 1 KN = 1, NMAX
# *
# * ..Some abbreviations
# *
#        N  = NA(KN)
#        S1 = S(KN,1)
#        S2 = S(KN,2)
#        S3 = S(KN,3)
# *
#        NM = N - 1.
#        N1 = N + 1.
#        N2 = N + 2.
#        NI = 1./N
#        NMI = 1./NM
#        N1I = 1./N1
#        N2I = 1./N2
# *
#        S1M = S1 - NI
#        S2M = S2 - NI*NI
#        S3M = S3 - NI**3
#        S11 = S1 + N1I
#        S21 = S2 + N1I*N1I
#        S31 = S3 + N1I**3
#        S22 = S21 + N2I*N2I
# *
# * ---------------------------------------------------------------------
# *
# *  ..Moments of the basic x-space functions 
# *
#        A0  = - S1M
# *
#        B1  = - S1 * NI
#        B1M = - S1M * NMI
#        B11 = - S11 * N1I
#        B2  = (S1**2 + S2) * NI
#        B2M = (S1M**2 + S2M) * NMI
#        B21 = (S11**2 + S21) * N1I
#        B3  = - (S1**3 + 3.*S1*S2 + 2.*S3) * NI
# *
#        C0 = NI
#        CM = NMI
#        C1 = N1I
#        C2 = N2I
# *
#        D1  = - NI*NI
#        D11 = - N1I*N1I
#        D12 = - N2I*N2I
#        D2  = 2.* NI**3
#        D21 = 2.* N1I**3
#        D22 = 2.* N2I**3
#        D3  = - 6.* NI**4
#        D31 = - 6.* N1I**4
# *
#        E2 = 2.* NI * ( ZETA(3) - S3 + NI * (ZETA(2) - S2 - NI * S1) )
# *
#        F1  = NI  * ( ZETA(2) - S2 )
#        F1M = NMI * ( ZETA(2) - S2M )
#        F11 = N1I * ( ZETA(2) - S21 )
#        F12 = N2I * ( ZETA(2) - S22 )
#        F2  = - NI  * F1
#        F21 = - N1I * F11
# *
# * ---------------------------------------------------------------------
# *             
# * ..The moments of the OME's A_Hq^{PS,(2)} and A_Hg^{S,(2)} given in 
# *    Eqs. (B.1) and (B.3) of BMSN. For the latter quantity an accurate
# *    x-space parametrization is used instead of the full expression.
# *
#        A2HQ = - (32./3.D0 * CM + 8.* (C0-C1) - 32./3.D0 * C2) * ZETA(2)
#      1        - 448./27.D0 * CM - 4./3.D0 * C0 - 124./3.D0 * C1 
#      2        + 1600./27.D0 * C2 - 4./3.D0 * (D3 + D31) + 2.* D2 
#      3        + 10.* D21 + 16./3.D0 * D22 - 16.* ZETA(2) * (D1 + D11) 
#      4        - 56./3.D0 * D1 - 88./3.D0 * D11 - 448./9.D0 * D12 
#      5        + 32./3.D0 * F1M + 8.* (F1 - F11) - 32./3.D0 * F12 
#      6        + 16.* (F2 + F21)
# *
#        A2HG = - 0.006 - 1.111 * B3 - 0.400 * B2 - 2.770 * B1
#      1        - 24.89 * CM - 187.8 * C0 + 249.6 * C1
#      2        - 1.556 * D3 - 3.292 * D2 - 93.68 * D1 - 146.8 * E2
# *
# * ..The moments of the OME's A_{gq,H}^{S,(2)} and A_{gg,H}^{S,(2)} 
# *    given in Eqs. (B.5) and (B.7) of BMSN.
# *
#        A2GQ =   4./3.D0 * (2.* B2M - 2.* B2 + B21)
#      1        + 8./9.D0 * (10.* B1M - 10.* B1 + 8.* B11)
#      2        + 1./27.D0 * (448.* (CM - C0) + 344.* C1)  
# *
#        A2GGF = - 15. - 8.* CM + 80.* C0 - 48.* C1 - 24.* C2 
#      1         + 4./3.D0 * (D3 + D31) + 6.* D2 + 10.* D21 + 32.* D1 
#      2         + 48.* D11
#        A2GGA =   224./27.D0 * A0 + 10./9.D0 - 4./3.D0 * B11 
#      1         + 1./27.D0 * (556.* CM - 628.* C0 + 548.* C1 - 700.* C2) 
#      2         + 4./3.D0 * (D2 + D21) + 1./9.D0 * (52.* D1 + 88.* D11)
# *
# * ---------------------------------------------------------------------
# *
# * ..Output to the array 
# *
#        A2SG(KN,1,1) = CF*TR * A2HQ
#        A2SG(KN,1,2) = A2HG
#        A2SG(KN,2,1) = CF*TR * A2GQ
#        A2SG(KN,2,2) = TR * (CF * A2GGF + CA * A2GGA)  
# *
# * ---------------------------------------------------------------------
# *
#   1    CONTINUE
# *
#        RETURN
#        END
# *
# * =================================================================av==
