*
* ..File: xkij1e.f    K_ij^(1)
* 
*
* ..The two-loop physical evolution kernel for the system (F_2, F_phi)  
*    of structure function at mu_r = Q, expanded in alpha_s/(4 pi).
*
* ..The distributions (in the mathematical sense) are given as in eq.
*    (B.26) of Floratos, Kounnas, Lacaze: Nucl. Phys. B192 (1981) 417.
*    The name-endings A, B, and C of the functions below correspond to
*    the kernel superscripts [2], [3], and [1] in that equation.
* 
* ..The code uses the package of Gehrmann and Remiddi for the harmonic
*    polylogarithms published in hep-ph/0107173 = CPC 141 (2001) 296.
*
* ..Reference: G. Soar, S. Moch, J. Vermaseren and A. Vogt,
*              arXiv:0912.0369 [hep-ph]
*
* =====================================================================
*
*
* ..K_22, regular part
*
       FUNCTION XK221A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4 
       INTEGER NF, N1, N2, NW, I1, I2, I3
       PARAMETER ( N1 = -1, N2 = 1, NW = 4 ) 
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2), 
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2) 
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2), 
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2) 
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2), 
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2) 
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
* ..The soft coefficient for use in XK222B and XK222C
*
       COMMON / K21SOFT / K21A1, K21A0
*
* ...Colour factors
*
       CF  = 4./3.D0
       CA  = 3.D0
*
* ...Some abbreviations
*
       DX = 1.D0/X
       DM = 1.D0/(1.D0-X)
       DP = 1.D0/(1.D0+X)
       DL1 = LOG (1.D0-X)
*
* ...The harmonic polylogs up to weight NW = 2 by Gehrmann and Remiddi
*
       CALL HPLOG (X, NW, HC1,HC2,HC3,HC4, HR1,HR2,HR3,HR4,
     ,            HI1,HI2,HI3,HI4, N1, N2) 
*
* ...The physical kernel in terms of the harmonic polylogs
*    (without the delta(1-x) part, but with the soft contribution)
*
      Kqq1 =
     &  + cf*ca * (  - 164.D0/9.D0 - 434.D0/9.D0*x + 367.D0/9.D0*dm + 8.
     &    D0*z2*x + 8.D0*z2*dp - 8.D0*z2*dm - 44.D0/3.D0*Hr1(0) - 44.D0/
     &    3.D0*Hr1(0)*x + 88.D0/3.D0*Hr1(0)*dm - 22.D0/3.D0*Hr1(1) - 22.
     &    D0/3.D0*Hr1(1)*x + 44.D0/3.D0*Hr1(1)*dm - 8.D0*Hr2(-1,0) + 8.D
     &    0*Hr2(-1,0)*x + 16.D0*Hr2(-1,0)*dp - 8.D0*Hr2(0,0)*x - 8.D0*
     &    Hr2(0,0)*dp + 8.D0*Hr2(0,0)*dm )
      Kqq1 = Kqq1 + cf**2 * (  - 4.D0 + 4.D0*x + 8.D0*z2 - 8.D0*z2*x -
     &    16.D0*z2*dp + 8.D0*Hr1(0) - 12.D0*Hr1(0)*dm + 16.D0*Hr2(-1,0)
     &     - 16.D0*Hr2(-1,0)*x - 32.D0*Hr2(-1,0)*dp - 12.D0*Hr2(0,0) +
     &    4.D0*Hr2(0,0)*x + 16.D0*Hr2(0,0)*dp - 8.D0*Hr2(0,1) - 8.D0*
     &    Hr2(0,1)*x + 16.D0*Hr2(0,1)*dm - 8.D0*Hr2(1,0) - 8.D0*Hr2(1,0
     &    )*x + 16.D0*Hr2(1,0)*dm )
      Kqq1 = Kqq1 + nf*cf * (  - 382.D0/9.D0 + 338.D0/9.D0*x - 16.D0/3.D
     &    0*x**2 + 64.D0/3.D0*dx - 58.D0/9.D0*dm + 8.D0/3.D0*Hr1(0) -
     &    28.D0/3.D0*Hr1(0)*x + 32.D0/3.D0*Hr1(0)*x**2 + 16.D0/3.D0*
     &    Hr1(0)*dx - 16.D0/3.D0*Hr1(0)*dm + 4.D0/3.D0*Hr1(1) + 4.D0/3.D
     &    0*Hr1(1)*x - 8.D0/3.D0*Hr1(1)*dm - 8.D0*Hr2(0,0) - 8.D0*Hr2(0
     &    ,0)*x )
*
* ...The soft (`+'-distribution) part of the kernel
*
       K21A1 =
     &    +8.D0/3.D0*cf*nf
     &    -44.D0/3.D0*cf*ca
       K21A0 =
     &    -58.D0/9.D0*cf*nf
     &    +367.D0/9.D0*cf*ca
     &    -8.D0*z2*cf*ca
*
       Kqq1L = DM* ( DL1* K21A1 + K21A0 )

* ...The regular piece of the kernel
*
       XK221A = Kqq1 - Kqq1L
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The singular (soft) piece.
*
       FUNCTION XK221B (Y, NF)
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       COMMON / K21SOFT / K21A1, K21A0
*
       DL1 = LOG (1.D0-Y)
       DM  = 1.D0/(1.D0-Y)
*
       XK221B  = DM* (DL1* K21A1 + K21A0 )
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The 'local' piece.
*
       FUNCTION XK221C (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
       COMMON / K21SOFT / K21A1, K21A0
*
* ...Colour factors
*
       CF  = 4./3.D0
       CA  = 3.D0
*
* ...The coefficient of delta(1-x)
*
       K21DELT = 
     &    -19.D0/3.D0*cf*nf
     &    +215.D0/6.D0*cf*ca
     &    +3.D0/2.D0*cf**2
     &    -12.D0*z3*cf*ca
     &    +24.D0*z3*cf**2
     &    -16.D0/3.D0*z2*cf*nf
     &    +88.D0/3.D0*z2*cf*ca
     &    -12.D0*z2*cf**2
*
       DL1 = LOG (1.D0-Y)
*
       XK221C = DL1**2 * K21A1/2.D0 + DL1 * K21A0 + K21DELT
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..K_2phi 
*
       FUNCTION XK2P1A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4
       INTEGER NF, N1, N2, NW, I1, I2, I3
       PARAMETER ( N1 = -1, N2 = 1, NW = 2 )
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2),
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2),
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2),
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2)
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
* ...Colour factors and an abbreviation
*
       CF  = 4./3.D0
       CA  = 3.D0
*
       DX = 1.D0/X
*
* ...The harmonic polylogs up to weight 2 by Gehrmann and Remiddi
*
       CALL HPLOG (X, NW, HC1,HC2,HC3,HC4, HR1,HR2,HR3,HR4,
     ,            HI1,HI2,HI3,HI4, N1, N2)
*
* ...The physical kernel in terms of the harmonic polylogs
*
      Kqg1 =
     &  + nf*ca * (  - 644.D0/9.D0 + 310.D0/9.D0*x + 4.D0*x**2 + 200.D0/
     &    9.D0*dx - 8.D0*z2 - 16.D0*z2*x**2 - 22.D0/3.D0*Hr1(0) - 196.D0
     &    /3.D0*Hr1(0)*x + 44.D0*Hr1(0)*x**2 + 16.D0/3.D0*Hr1(0)*dx + 2.
     &    D0/3.D0*Hr1(1) - 100.D0/3.D0*Hr1(1)*x + 100.D0/3.D0*Hr1(1)*
     &    x**2 - 8.D0*Hr2(-1,0) - 16.D0*Hr2(-1,0)*x - 16.D0*Hr2(-1,0)*
     &    x**2 - 8.D0*Hr2(0,0) - 16.D0*Hr2(0,0)*x + 8.D0*Hr2(0,1) - 16.D
     &    0*Hr2(0,1)*x + 16.D0*Hr2(0,1)*x**2 )
      Kqg1 = Kqg1 + nf*cf * ( 4.D0 - 14.D0*x + 8.D0*x**2 + 2.D0*Hr1(0)
     &     + 8.D0*Hr1(0)*x - 24.D0*Hr1(0)*x**2 + 4.D0*Hr1(1) + 24.D0*
     &    Hr1(1)*x - 24.D0*Hr1(1)*x**2 + 4.D0*Hr2(0,0) - 8.D0*Hr2(0,0)*
     &    x + 16.D0*Hr2(0,0)*x**2 + 8.D0*Hr2(1,0) - 16.D0*Hr2(1,0)*x +
     &    16.D0*Hr2(1,0)*x**2 )
      Kqg1 = Kqg1 + nf**2 * ( 44.D0/9.D0 - 100.D0/9.D0*x + 28.D0/3.D0*
     &    x**2 - 8.D0/9.D0*dx + 4.D0/3.D0*Hr1(0) - 8.D0/3.D0*Hr1(0)*x
     &     + 8.D0/3.D0*Hr1(0)*x**2 + 4.D0/3.D0*Hr1(1) - 8.D0/3.D0*Hr1(1
     &    )*x + 8.D0/3.D0*Hr1(1)*x**2 )
*
       XK2P1A = Kqg1
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..K_phi 2 
*
       FUNCTION XKP21A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4
       INTEGER NF, N1, N2, NW, I1, I2, I3
       PARAMETER ( N1 = -1, N2 = 1, NW = 2 )
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2),
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2),
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2),
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2)
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0,
     ,             Z4 = 1.0823 23233 71113 81916 D0 )
*
* ...Colour factors and an abbreviation
*
       CF  = 4./3.D0
       CA  = 3.D0
*
       DX = 1.D0/X
*
* ...The harmonic polylogs up to weight 2 by Gehrmann and Remiddi
*
       CALL HPLOG (X, NW, HC1,HC2,HC3,HC4, HR1,HR2,HR3,HR4,
     ,            HI1,HI2,HI3,HI4, N1, N2)
*
* ...The physical kernel in terms of the harmonic polylogs
*
      Kgq1 =
     &  + cf*ca * (  - 22.D0/3.D0 + 83.D0/3.D0*x + 64.D0/3.D0*x**2 + 29.
     &    D0*dx + 8.D0*z2*x + 16.D0*z2*dx - 92.D0*Hr1(0) + 10.D0*Hr1(0)
     &    *x - 32.D0/3.D0*Hr1(0)*x**2 + 44.D0/3.D0*Hr1(0)*dx - 44.D0/3.D
     &    0*Hr1(1) + 22.D0/3.D0*Hr1(1)*x + 8.D0/3.D0*Hr1(1)*dx + 16.D0*
     &    Hr2(-1,0) + 8.D0*Hr2(-1,0)*x + 16.D0*Hr2(-1,0)*dx + 16.D0*
     &    Hr2(0,0) + 8.D0*Hr2(0,0)*x - 16.D0*Hr2(1,0) + 8.D0*Hr2(1,0)*x
     &     + 16.D0*Hr2(1,0)*dx )
      Kgq1 = Kgq1 + cf**2 * (  - 18.D0 - x + 10.D0*dx + 16.D0*z2 - 8.D0
     &    *z2*x - 16.D0*z2*dx + 4.D0*Hr1(0) + 6.D0*Hr1(0)*x + 12.D0*
     &    Hr1(1)*dx - 8.D0*Hr2(0,0) + 4.D0*Hr2(0,0)*x - 16.D0*Hr2(0,1)
     &     + 8.D0*Hr2(0,1)*x + 16.D0*Hr2(0,1)*dx )
      Kgq1 = Kgq1 + nf*cf * ( 52.D0/3.D0 - 26.D0/3.D0*x - 58.D0/3.D0*dx
     &     + 8.D0*Hr1(0) - 4.D0*Hr1(0)*x - 8.D0*Hr1(0)*dx + 8.D0/3.D0*
     &    Hr1(1) - 4.D0/3.D0*Hr1(1)*x - 8.D0/3.D0*Hr1(1)*dx )
*
       XKP21A = Kgq1
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..K_phi phi, regular part
*
       FUNCTION XKPP1A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4 
       INTEGER NF, N1, N2, NW, I1, I2, I3
       PARAMETER ( N1 = -1, N2 = 1, NW = 2 ) 
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2), 
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2) 
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2), 
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2) 
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2), 
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2) 
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
* ..The soft coefficient for use in XKPP1B and XKPP1C
*
       COMMON / KP1SOFT / KP1A1, KP1A0
*
* ...Colour factors
*
       CF  = 4./3.D0
       CA  = 3.D0
*
* ...Some abbreviations
*
       DX = 1.D0/X
       DM = 1.D0/(1.D0-X)
       DP = 1.D0/(1.D0+X)
       DL1 = LOG (1.D0-X)
*
* ...The harmonic polylogs up to weight NW = 2 by Gehrmann and Remiddi
*
       CALL HPLOG (X, NW, HC1,HC2,HC3,HC4, HR1,HR2,HR3,HR4,
     ,            HI1,HI2,HI3,HI4, N1, N2) 
*
* ...The physical kernel in terms of the harmonic polylogs
*    (without the delta(1-x) part, but with the soft contribution)
*
      Kgg1 =
     &  + ca**2 * (  - 50.D0/9.D0 - 218.D0/9.D0*x + 121.D0/9.D0*dx +
     &    389.D0/9.D0*dm + 32.D0*z2 + 16.D0*z2*x**2 - 8.D0*z2*dp - 8.D0
     &    *z2*dm - 188.D0/3.D0*Hr1(0) + 88.D0/3.D0*Hr1(0)*x - 220.D0/3.D
     &    0*Hr1(0)*x**2 + 44.D0/3.D0*Hr1(0)*dx + 44.D0/3.D0*Hr1(0)*dm
     &     - 88.D0/3.D0*Hr1(1) + 44.D0/3.D0*Hr1(1)*x - 44.D0/3.D0*Hr1(1
     &    )*x**2 + 44.D0/3.D0*Hr1(1)*dx + 44.D0/3.D0*Hr1(1)*dm + 32.D0*
     &    Hr2(-1,0) + 16.D0*Hr2(-1,0)*x + 16.D0*Hr2(-1,0)*x**2 + 16.D0*
     &    Hr2(-1,0)*dx - 16.D0*Hr2(-1,0)*dp + 32.D0*Hr2(0,0)*x - 16.D0*
     &    Hr2(0,0)*x**2 + 8.D0*Hr2(0,0)*dp + 8.D0*Hr2(0,0)*dm - 32.D0*
     &    Hr2(0,1) + 16.D0*Hr2(0,1)*x - 16.D0*Hr2(0,1)*x**2 + 16.D0*
     &    Hr2(0,1)*dx + 16.D0*Hr2(0,1)*dm - 32.D0*Hr2(1,0) + 16.D0*Hr2(
     &    1,0)*x - 16.D0*Hr2(1,0)*x**2 + 16.D0*Hr2(1,0)*dx + 16.D0*Hr2(
     &    1,0)*dm )
      Kgg1 = Kgg1 + nf*ca * ( 116.D0/9.D0 - 76.D0/9.D0*x + 92.D0/9.D0*
     &    x**2 - 136.D0/9.D0*dx - 28.D0/3.D0*dm + 8.D0/3.D0*Hr1(0) - 16.
     &    D0/3.D0*Hr1(0)*x + 8.D0/3.D0*Hr1(0)*x**2 - 8.D0/3.D0*Hr1(0)*
     &    dx - 8.D0/3.D0*Hr1(0)*dm + 16.D0/3.D0*Hr1(1) - 8.D0/3.D0*Hr1(
     &    1)*x + 8.D0/3.D0*Hr1(1)*x**2 - 8.D0/3.D0*Hr1(1)*dx - 8.D0/3.D0
     &    *Hr1(1)*dm )
      Kgg1 = Kgg1 + nf*cf * ( 6.D0 + 10.D0*x - 56.D0/9.D0*x**2 - 88.D0/
     &    9.D0*dx - 8.D0*Hr1(0) + 12.D0*Hr1(0)*x - 16.D0/3.D0*Hr1(0)*dx
     &     - 8.D0*Hr2(0,0) - 8.D0*Hr2(0,0)*x )
      Kgg1 = Kgg1 + nf**2 * ( 4.D0/9.D0*dx + 4.D0/9.D0*dm )
*
* ...The soft (`+'-distribution) part of the kernel
*
       KP1A1 =
     &    +8.D0/3.D0*ca*nf
     &    -44.D0/3.D0*ca**2
       KP1A0 =
     &    +4.D0/9.D0*nf**2
     &    -28.D0/3.D0*ca*nf
     &    +389.D0/9.D0*ca**2
     &    -8.D0*z2*ca**2
*
       Kgg1L = DM* ( DL1* KP1A1 + KP1A0 )

* ...The regular piece of the kernel
*
       XKPP1A = Kgg1 - Kgg1L
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The singular (soft) piece.
*
       FUNCTION XKPP1B (Y, NF)
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       COMMON / KP1SOFT / KP1A1, KP1A0
*
       DL1 = LOG (1.D0-Y)
       DM  = 1.D0/(1.D0-Y)
*
       XKPP1B  = DM* ( DL1* KP1A1 + KP1A0 )
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The 'local' piece.
*
       FUNCTION XKPP1C (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
       COMMON / KP1SOFT / KP1A1, KP1A0
*
* ...Colour factors
*
       CF  = 4./3.D0
       CA  = 3.D0
*
* ...The coefficient of delta(1-x)
*
       KP1DELT = 
     &    -20.D0/27.D0*nf**2
     &    +172.D0/27.D0*ca*nf
     &    -449.D0/27.D0*ca**2
     &    -2.D0*cf*nf
     &    +12.D0*z3*ca**2
     &    -8.D0/3.D0*z2*ca*nf
     &    +44.D0/3.D0*z2*ca**2
*
       DL1 = LOG (1.D0-Y)
*
       XKPP1C = DL1**2 * KP1A1/2.D0 + DL1 * KP1A0 + KP1DELT
*
       RETURN
       END
*
* =================================================================av==
