*
* ..File: xcpqg2e.f    F_phi,Q and F_phi,G
*
*
* ..The 2-loop MS(bar) quark and gluon coefficient functions for the
*    structure function F_phi in deep-inelastic scattering at mu_f = Q.
*    The expansion parameter is alpha_s/(4 pi).
* 
* ..The distributions (in the mathematical sense) are given as in eq.
*    (B.26) of Floratos, Kounnas, Lacaze: Nucl. Phys. B192 (1981) 417.
*    The name-endings A, B, and C of the functions below correspond to
*    the kernel superscripts [2], [3], and [1] in that equation.
*
* ..The code uses the package of Gehrmann and Remiddi for the harmonic
*    polylogarithms published in hep-ph/0107173 = CPC 141 (2001) 296,
*    upgraded to weight 5 (T. Gehrmann, private communication).
*
* ..Reference: G. Soar, S. Moch, J. Vermaseren and A. Vogt,
*              arXiv:0912.0369 [hep-ph]
*
* =====================================================================
*
*
* ..The two-loop quark coefficient function 
*
       FUNCTION CPHIQ2A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4, HC5
       INTEGER NF, N1, N2, NW
       PARAMETER ( N1 = -1, N2 = 1, NW = 3 )
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2),
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HC5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2),
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HR5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2),
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HI5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
* ...Colour factors and abbreviations
*
       CF  = 4./3.D0
       CA  = 3.D0
*
       DX = 1.D0/X
*
* ...Harmonic polylogs (HPLs) up to weight NW=3 by Gehrmann and Remiddi 
*
       CALL HPLOG5 (X, NW, HC1,HC2,HC3,HC4,HC5, HR1,HR2,HR3,HR4,HR5,
     ,             HI1,HI2,HI3,HI4,HI5, N1, N2)
*
* ...The coefficient function in terms of the HPLs
*
       cphiq2 =
     &  + cf*ca * ( 3086.D0/27.D0 + 38.D0/27.D0*x + 608.D0/27.D0*x**2
     &     - 13457.D0/54.D0*dx - 64.D0*z3 + 44.D0*z3*x + 72.D0*z3*dx - 
     &    352.D0/3.D0*z2 - 28.D0/3.D0*z2*x - 16.D0*z2*x**2 + 212.D0/3.D0
     &    *z2*dx + 16.D0*Hr1(-1)*z2 + 8.D0*Hr1(-1)*z2*x + 16.D0*Hr1(-1)
     &    *z2*dx - 20.D0/3.D0*Hr1(0) - 57.D0*Hr1(0)*x - 352.D0/9.D0*
     &    Hr1(0)*x**2 - 454.D0/3.D0*Hr1(0)*dx + 16.D0*Hr1(0)*z2 + 32.D0
     &    *Hr1(0)*z2*x + 32.D0*Hr1(0)*z2*dx + 242.D0/9.D0*Hr1(1) - 127.D
     &    0/9.D0*Hr1(1)*x - 176.D0/9.D0*Hr1(1)*x**2 - 32.D0*Hr1(1)*dx
     &     - 40.D0*Hr1(1)*z2 + 20.D0*Hr1(1)*z2*x + 40.D0*Hr1(1)*z2*dx
     &     + 32.D0*Hr2(-1,0) + 12.D0*Hr2(-1,0)*x + 16.D0/3.D0*Hr2(-1,0)
     &    *x**2 + 88.D0/3.D0*Hr2(-1,0)*dx + 536.D0/3.D0*Hr2(0,0) + 26.D0
     &    /3.D0*Hr2(0,0)*x + 64.D0/3.D0*Hr2(0,0)*x**2 - 212.D0/3.D0*
     &    Hr2(0,0)*dx + 352.D0/3.D0*Hr2(0,1) + 64.D0/3.D0*Hr2(0,1)*x + 
     &    16.D0*Hr2(0,1)*x**2 - 124.D0/3.D0*Hr2(0,1)*dx + 184.D0/3.D0*
     &    Hr2(1,0) + 4.D0/3.D0*Hr2(1,0)*x + 16.D0/3.D0*Hr2(1,0)*x**2 - 
     &    212.D0/3.D0*Hr2(1,0)*dx )
      cphiq2 = cphiq2 + cf*ca * ( 140.D0/3.D0*Hr2(1,1) + 26.D0/3.D0*
     &    Hr2(1,1)*x + 16.D0/3.D0*Hr2(1,1)*x**2 - 56.D0*Hr2(1,1)*dx - 
     &    24.D0*Hr3(-1,0,0) - 12.D0*Hr3(-1,0,0)*x - 24.D0*Hr3(-1,0,0)*
     &    dx - 16.D0*Hr3(-1,0,1) - 8.D0*Hr3(-1,0,1)*x - 16.D0*Hr3(-1,0,
     &    1)*dx - 16.D0*Hr3(0,-1,0) - 40.D0*Hr3(0,0,0) - 28.D0*Hr3(0,0,
     &    0)*x - 16.D0*Hr3(0,0,0)*dx - 16.D0*Hr3(0,0,1) - 32.D0*Hr3(0,0
     &    ,1)*x - 32.D0*Hr3(0,0,1)*dx - 24.D0*Hr3(0,1,0)*x - 32.D0*Hr3(
     &    0,1,0)*dx + 8.D0*Hr3(0,1,1) - 28.D0*Hr3(0,1,1)*x - 40.D0*Hr3(
     &    0,1,1)*dx + 40.D0*Hr3(1,0,0) - 20.D0*Hr3(1,0,0)*x - 40.D0*
     &    Hr3(1,0,0)*dx + 40.D0*Hr3(1,0,1) - 20.D0*Hr3(1,0,1)*x - 40.D0
     &    *Hr3(1,0,1)*dx + 40.D0*Hr3(1,1,0) - 20.D0*Hr3(1,1,0)*x - 40.D0
     &    *Hr3(1,1,0)*dx + 40.D0*Hr3(1,1,1) - 20.D0*Hr3(1,1,1)*x - 40.D0
     &    *Hr3(1,1,1)*dx )
      cphiq2 = cphiq2 + cf**2 * ( 104.D0 - 15.D0*x - 59.D0/2.D0*dx - 8.D
     &    0*z3 - 12.D0*z3*x - 32.D0*z3*dx - 20.D0*z2 + 8.D0*z2*x + 36.D0
     &    *z2*dx + 16.D0*Hr1(-1)*z2 + 8.D0*Hr1(-1)*z2*x + 16.D0*Hr1(-1)
     &    *z2*dx + 15.D0*Hr1(0) + 18.D0*Hr1(0)*x - 10.D0*Hr1(0)*dx - 32.
     &    D0*Hr1(0)*z2 + 16.D0*Hr1(0)*z2*x + 16.D0*Hr1(0)*z2*dx + 56.D0
     &    *Hr1(1) + 3.D0*Hr1(1)*x - 64.D0*Hr1(1)*dx - 16.D0*Hr2(-1,0)
     &     - 16.D0*Hr2(-1,0)*x - 12.D0*Hr2(0,0) + Hr2(0,0)*x + 20.D0*
     &    Hr2(0,1) - 24.D0*Hr2(0,1)*x - 36.D0*Hr2(0,1)*dx + 32.D0*Hr2(1
     &    ,0) - 14.D0*Hr2(1,0)*x - 36.D0*Hr2(1,0)*dx + 44.D0*Hr2(1,1)
     &     - 24.D0*Hr2(1,1)*x - 48.D0*Hr2(1,1)*dx + 32.D0*Hr3(-1,-1,0)
     &     + 16.D0*Hr3(-1,-1,0)*x + 32.D0*Hr3(-1,-1,0)*dx - 16.D0*Hr3(
     &    -1,0,0) - 8.D0*Hr3(-1,0,0)*x - 16.D0*Hr3(-1,0,0)*dx - 32.D0*
     &    Hr3(0,-1,0) + 20.D0*Hr3(0,0,0) - 10.D0*Hr3(0,0,0)*x + 32.D0*
     &    Hr3(0,0,1) - 16.D0*Hr3(0,0,1)*x - 16.D0*Hr3(0,0,1)*dx + 24.D0
     &    *Hr3(0,1,0) - 12.D0*Hr3(0,1,0)*x - 16.D0*Hr3(0,1,0)*dx + 24.D0
     &    *Hr3(0,1,1) )
      cphiq2 = cphiq2 + cf**2 * (  - 12.D0*Hr3(0,1,1)*x - 16.D0*Hr3(0,1
     &    ,1)*dx + 16.D0*Hr3(1,0,1) - 8.D0*Hr3(1,0,1)*x - 16.D0*Hr3(1,0
     &    ,1)*dx + 16.D0*Hr3(1,1,0) - 8.D0*Hr3(1,1,0)*x - 16.D0*Hr3(1,1
     &    ,0)*dx + 8.D0*Hr3(1,1,1) - 4.D0*Hr3(1,1,1)*x - 8.D0*Hr3(1,1,1
     &    )*dx )
      cphiq2 = cphiq2 + nf*cf * (  - 332.D0/27.D0 + 28.D0/27.D0*x + 737.
     &    D0/27.D0*dx + 16.D0/3.D0*z2 - 8.D0/3.D0*z2*x - 16.D0/3.D0*z2*
     &    dx - 52.D0/3.D0*Hr1(0) + 22.D0/3.D0*Hr1(0)*x + 64.D0/3.D0*
     &    Hr1(0)*dx - 116.D0/9.D0*Hr1(1) + 58.D0/9.D0*Hr1(1)*x + 116.D0/
     &    9.D0*Hr1(1)*dx - 32.D0/3.D0*Hr2(0,0) + 16.D0/3.D0*Hr2(0,0)*x
     &     + 32.D0/3.D0*Hr2(0,0)*dx - 16.D0/3.D0*Hr2(0,1) + 8.D0/3.D0*
     &    Hr2(0,1)*x + 16.D0/3.D0*Hr2(0,1)*dx - 16.D0/3.D0*Hr2(1,0) + 8.
     &    D0/3.D0*Hr2(1,0)*x + 16.D0/3.D0*Hr2(1,0)*dx - 8.D0/3.D0*Hr2(1
     &    ,1) + 4.D0/3.D0*Hr2(1,1)*x + 8.D0/3.D0*Hr2(1,1)*dx )
* 
       CPHIQ2A = cphiq2
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The regular piece of the gluon coefficient function
*
       FUNCTION CPHIG2A (X, NF)
*
       IMPLICIT REAL*8 (A - Z)
       COMPLEX*16 HC1, HC2, HC3, HC4, HC5
       INTEGER NF, NF2, N1, N2, NW
       PARAMETER ( N1 = -1, N2 = 1, NW = 3 )
       DIMENSION HC1(N1:N2),HC2(N1:N2,N1:N2),HC3(N1:N2,N1:N2,N1:N2),
     ,           HC4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HC5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HR1(N1:N2),HR2(N1:N2,N1:N2),HR3(N1:N2,N1:N2,N1:N2),
     ,           HR4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HR5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       DIMENSION HI1(N1:N2),HI2(N1:N2,N1:N2),HI3(N1:N2,N1:N2,N1:N2),
     ,           HI4(N1:N2,N1:N2,N1:N2,N1:N2),
     ,           HI5(N1:N2,N1:N2,N1:N2,N1:N2,N1:N2)
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0 )
*
* ...The soft-gluon coefficients for use in CPHIG2B and CPHIG2C
*
       COMMON / CP2SOFT / CP2A0, CP2A1, CP2A2, CP2A3
*
* ...Colour factors and abbreviations
*
       CF  = 4./3.D0
       CA  = 3.D0
*
       DX = 1.D0/X
       DM = 1.D0/(1.D0-X)
       DP = 1.D0/(1.D0+X)
       DL1 = LOG (1.D0-X)
*
* ...Harmonic polylogs (HPLs) up to weight NW=3 by Gehrmann and Remiddi 
*
       CALL HPLOG5 (X, NW, HC1,HC2,HC3,HC4,HC5, HR1,HR2,HR3,HR4,HR5,
     ,             HI1,HI2,HI3,HI4,HI5, N1, N2)
*
* ...The coefficient function in terms of the harmonic polylogs
*    (without the delta(1-x) part, but with the soft contribution)
*
       cphig2 =
     &  + ca**2 * ( 5150.D0/27.D0 - 4486.D0/27.D0*x + 5747.D0/27.D0*
     &    x**2 - 2743.D0/9.D0*dx - 2570.D0/27.D0*dm - 160.D0*z3 + 72.D0
     &    *z3*x - 96.D0*z3*x**2 + 40.D0*z3*dx + 28.D0*z3*dp + 68.D0*z3*
     &    dm - 532.D0/3.D0*z2 + 272.D0/3.D0*z2*x - 484.D0/3.D0*z2*x**2
     &     + 352.D0/3.D0*z2*dx + 176.D0/3.D0*z2*dm + 64.D0*Hr1(-1)*z2
     &     + 32.D0*Hr1(-1)*z2*x + 32.D0*Hr1(-1)*z2*x**2 + 32.D0*Hr1(-1)
     &    *z2*dx - 32.D0*Hr1(-1)*z2*dp + 449.D0/9.D0*Hr1(0) - 1477.D0/9.
     &    D0*Hr1(0)*x + 242.D0/9.D0*Hr1(0)*x**2 - 1556.D0/9.D0*Hr1(0)*
     &    dx - 778.D0/9.D0*Hr1(0)*dm - 64.D0*Hr1(0)*z2 + 112.D0*Hr1(0)*
     &    z2*x - 64.D0*Hr1(0)*z2*x**2 + 48.D0*Hr1(0)*z2*dx + 8.D0*Hr1(0
     &    )*z2*dp + 56.D0*Hr1(0)*z2*dm + 1192.D0/9.D0*Hr1(1) - 662.D0/9.
     &    D0*Hr1(1)*x + 778.D0/9.D0*Hr1(1)*x**2 - 340.D0/3.D0*Hr1(1)*dx
     &     - 778.D0/9.D0*Hr1(1)*dm - 80.D0*Hr1(1)*z2 + 40.D0*Hr1(1)*z2*
     &    x - 40.D0*Hr1(1)*z2*x**2 + 40.D0*Hr1(1)*z2*dx + 40.D0*Hr1(1)*
     &    z2*dm + 16.D0*Hr2(-1,0) + 16.D0*Hr2(-1,0)*x + 88.D0/3.D0*Hr2(
     &    -1,0)*x**2 )
      cphig2 = cphig2 + ca**2 * ( 88.D0/3.D0*Hr2(-1,0)*dx + 202.D0*Hr2(
     &    0,0) - 66.D0*Hr2(0,0)*x + 572.D0/3.D0*Hr2(0,0)*x**2 - 220.D0/
     &    3.D0*Hr2(0,0)*dx - 44.D0*Hr2(0,0)*dm + 532.D0/3.D0*Hr2(0,1)
     &     - 224.D0/3.D0*Hr2(0,1)*x + 484.D0/3.D0*Hr2(0,1)*x**2 - 88.D0
     &    *Hr2(0,1)*dx - 176.D0/3.D0*Hr2(0,1)*dm + 136.D0*Hr2(1,0) - 92.
     &    D0*Hr2(1,0)*x + 308.D0/3.D0*Hr2(1,0)*x**2 - 352.D0/3.D0*Hr2(1
     &    ,0)*dx - 176.D0/3.D0*Hr2(1,0)*dm + 136.D0*Hr2(1,1) - 92.D0*
     &    Hr2(1,1)*x + 308.D0/3.D0*Hr2(1,1)*x**2 - 352.D0/3.D0*Hr2(1,1)
     &    *dx - 176.D0/3.D0*Hr2(1,1)*dm + 64.D0*Hr3(-1,-1,0) + 32.D0*
     &    Hr3(-1,-1,0)*x + 32.D0*Hr3(-1,-1,0)*x**2 + 32.D0*Hr3(-1,-1,0)
     &    *dx - 32.D0*Hr3(-1,-1,0)*dp - 80.D0*Hr3(-1,0,0) - 40.D0*Hr3(
     &    -1,0,0)*x - 40.D0*Hr3(-1,0,0)*x**2 - 40.D0*Hr3(-1,0,0)*dx + 
     &    40.D0*Hr3(-1,0,0)*dp - 32.D0*Hr3(-1,0,1) - 16.D0*Hr3(-1,0,1)*
     &    x - 16.D0*Hr3(-1,0,1)*x**2 - 16.D0*Hr3(-1,0,1)*dx + 16.D0*
     &    Hr3(-1,0,1)*dp - 96.D0*Hr3(0,-1,0) - 48.D0*Hr3(0,-1,0)*x**2
     &     + 24.D0*Hr3(0,-1,0)*dp )
      cphig2 = cphig2 + ca**2 * ( 24.D0*Hr3(0,-1,0)*dm - 96.D0*Hr3(0,0,
     &    0)*x + 40.D0*Hr3(0,0,0)*x**2 - 16.D0*Hr3(0,0,0)*dx - 12.D0*
     &    Hr3(0,0,0)*dp - 28.D0*Hr3(0,0,0)*dm + 64.D0*Hr3(0,0,1) - 112.D
     &    0*Hr3(0,0,1)*x + 64.D0*Hr3(0,0,1)*x**2 - 48.D0*Hr3(0,0,1)*dx
     &     - 8.D0*Hr3(0,0,1)*dp - 56.D0*Hr3(0,0,1)*dm + 64.D0*Hr3(0,1,0
     &    ) - 80.D0*Hr3(0,1,0)*x + 48.D0*Hr3(0,1,0)*x**2 - 48.D0*Hr3(0,
     &    1,0)*dx - 48.D0*Hr3(0,1,0)*dm + 80.D0*Hr3(0,1,1) - 88.D0*Hr3(
     &    0,1,1)*x + 56.D0*Hr3(0,1,1)*x**2 - 56.D0*Hr3(0,1,1)*dx - 56.D0
     &    *Hr3(0,1,1)*dm + 80.D0*Hr3(1,0,0) - 40.D0*Hr3(1,0,0)*x + 40.D0
     &    *Hr3(1,0,0)*x**2 - 40.D0*Hr3(1,0,0)*dx - 40.D0*Hr3(1,0,0)*dm
     &     + 112.D0*Hr3(1,0,1) - 56.D0*Hr3(1,0,1)*x + 56.D0*Hr3(1,0,1)*
     &    x**2 - 56.D0*Hr3(1,0,1)*dx - 56.D0*Hr3(1,0,1)*dm + 112.D0*
     &    Hr3(1,1,0) - 56.D0*Hr3(1,1,0)*x + 56.D0*Hr3(1,1,0)*x**2 - 56.D
     &    0*Hr3(1,1,0)*dx - 56.D0*Hr3(1,1,0)*dm + 96.D0*Hr3(1,1,1) - 48.
     &    D0*Hr3(1,1,1)*x + 48.D0*Hr3(1,1,1)*x**2 - 48.D0*Hr3(1,1,1)*dx
     &     - 48.D0*Hr3(1,1,1)*dm )
      cphig2 = cphig2 + nf*ca * (  - 566.D0/27.D0 + 580.D0/27.D0*x - 
     &    554.D0/27.D0*x**2 + 1312.D0/27.D0*dx + 224.D0/9.D0*dm + 40.D0/
     &    3.D0*z2 - 32.D0/3.D0*z2*x + 8.D0*z2*x**2 - 32.D0/3.D0*z2*dx
     &     - 32.D0/3.D0*z2*dm - 242.D0/9.D0*Hr1(0) + 196.D0/9.D0*Hr1(0)
     &    *x - 176.D0/9.D0*Hr1(0)*x**2 + 88.D0/3.D0*Hr1(0)*dx + 56.D0/3.
     &    D0*Hr1(0)*dm - 256.D0/9.D0*Hr1(1) + 182.D0/9.D0*Hr1(1)*x - 
     &    176.D0/9.D0*Hr1(1)*x**2 + 88.D0/3.D0*Hr1(1)*dx + 56.D0/3.D0*
     &    Hr1(1)*dm - 12.D0*Hr2(0,0) + 12.D0*Hr2(0,0)*x - 8.D0*Hr2(0,0)
     &    *x**2 + 8.D0*Hr2(0,0)*dx + 8.D0*Hr2(0,0)*dm - 40.D0/3.D0*Hr2(
     &    0,1) + 32.D0/3.D0*Hr2(0,1)*x - 8.D0*Hr2(0,1)*x**2 + 32.D0/3.D0
     &    *Hr2(0,1)*dx + 32.D0/3.D0*Hr2(0,1)*dm - 16.D0*Hr2(1,0) + 8.D0
     &    *Hr2(1,0)*x - 8.D0*Hr2(1,0)*x**2 + 32.D0/3.D0*Hr2(1,0)*dx + 
     &    32.D0/3.D0*Hr2(1,0)*dm - 16.D0*Hr2(1,1) + 8.D0*Hr2(1,1)*x - 8.
     &    D0*Hr2(1,1)*x**2 + 32.D0/3.D0*Hr2(1,1)*dx + 32.D0/3.D0*Hr2(1,
     &    1)*dm )
      cphig2 = cphig2 + nf*cf * ( 521.D0/9.D0 - 386.D0/9.D0*x - 496.D0/
     &    27.D0*x**2 + 361.D0/27.D0*dx + 2.D0*dm - 8.D0*z3 - 8.D0*z3*x
     &     - 12.D0*z2 - 8.D0*z2*x + 16.D0/3.D0*z2*x**2 - 16.D0/3.D0*z2*
     &    dx + 86.D0/3.D0*Hr1(0) - 2.D0/3.D0*Hr1(0)*x - 136.D0/9.D0*
     &    Hr1(0)*x**2 + 64.D0/9.D0*Hr1(0)*dx - 16.D0*Hr1(0)*z2 - 16.D0*
     &    Hr1(0)*z2*x + 62.D0/3.D0*Hr1(1) - 38.D0/3.D0*Hr1(1)*x - 136.D0
     &    /9.D0*Hr1(1)*x**2 + 64.D0/9.D0*Hr1(1)*dx + 14.D0*Hr2(0,0) + 
     &    10.D0*Hr2(0,0)*x - 16.D0/3.D0*Hr2(0,0)*x**2 + 16.D0/3.D0*Hr2(
     &    0,0)*dx + 12.D0*Hr2(0,1) + 8.D0*Hr2(0,1)*x - 16.D0/3.D0*Hr2(0
     &    ,1)*x**2 + 16.D0/3.D0*Hr2(0,1)*dx + 4.D0*Hr2(1,0) - 4.D0*Hr2(
     &    1,0)*x - 16.D0/3.D0*Hr2(1,0)*x**2 + 16.D0/3.D0*Hr2(1,0)*dx + 
     &    4.D0*Hr2(1,1) - 4.D0*Hr2(1,1)*x - 16.D0/3.D0*Hr2(1,1)*x**2 + 
     &    16.D0/3.D0*Hr2(1,1)*dx + 20.D0*Hr3(0,0,0) + 20.D0*Hr3(0,0,0)*
     &    x + 16.D0*Hr3(0,0,1) + 16.D0*Hr3(0,0,1)*x + 8.D0*Hr3(0,1,0)
     &     + 8.D0*Hr3(0,1,0)*x + 8.D0*Hr3(0,1,1) + 8.D0*Hr3(0,1,1)*x )
      cphig2 = cphig2 + nf**2 * (  - 40.D0/27.D0*dx - 40.D0/27.D0*dm - 
     &    8.D0/9.D0*Hr1(0)*dx - 8.D0/9.D0*Hr1(0)*dm - 8.D0/9.D0*Hr1(1)*
     &    dx - 8.D0/9.D0*Hr1(1)*dm )
*
* ...The soft (`+'-distribution) part of the coefficient function
*
       CP2A3 = 
     &    +8.D0*ca**2
       CP2A2 = 
     &    +16.D0/3.D0*ca*nf
     &    -88.D0/3.D0*ca**2
       CP2A1 =
     &    +8.D0/9.D0*nf**2
     &    -56.D0/3.D0*ca*nf
     &    +778.D0/9.D0*ca**2
     &    -40.D0*z2*ca**2
       CP2A0 = 
     &    -40.D0/27.D0*nf**2
     &    +224.D0/9.D0*ca*nf
     &    -2570.D0/27.D0*ca**2
     &    +2.D0*cf*nf
     &    +32.D0*z3*ca**2
     &    -32.D0/3.D0*z2*ca*nf
     &    +176.D0/3.D0*z2*ca**2
*
       CPHIG2L = DM* ( DL1**3* CP2A3 + DL1**2* CP2A2 
     1               + DL1* CP2A1    + CP2A0 )

* ...The regular piece of the coefficient function
*
       CPHIG2A = cphig2 - CPHIG2L
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The singular (soft) piece 
*
       FUNCTION CPHIG2B (Y, NF)
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       COMMON / CP2SOFT / CP2A0, CP2A1, CP2A2, CP2A3
*
       DL1 = LOG (1.D0-Y)
       DM  = 1.D0/(1.D0-Y)
*
       CPHIG2B = DM* ( DL1**3* CP2A3 + DL1**2* CP2A2 
     1               + DL1*    CP2A1 + CP2A0 ) 
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The 'local' piece 
*
       FUNCTION CPHIG2C (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0,
     ,             Z3 = 1.2020 56903 15959 42854 D0)
*
       COMMON / CP2SOFT / CP2A0, CP2A1, CP2A2, CP2A3
*
* ...Colour factors
*
       CF  = 4./3.D0
       CA  = 3.D0
*
* ...The coefficient of delta(1-x)
*
       C2DELT =
     &    +100.D0/81.D0*nf**2
     &    -4112.D0/81.D0*ca*nf
     &    +30425.D0/162.D0*ca**2
     &    -63.D0/2.D0*cf*nf
     &    -28.D0/3.D0*z3*ca*nf
     &    -242.D0/3.D0*z3*ca**2
     &    +24.D0*z3*cf*nf
     &    -8.D0/9.D0*z2*nf**2
     &    +56.D0/3.D0*z2*ca*nf
     &    -778.D0/9.D0*z2*ca**2
     &    +101.D0/5.D0*z2**2*ca**2
*
       DL1 = LOG (1.D0-Y)
*
       CPHIG2C =  DL1**4 * CP2A3/4.D0 + DL1**3 * CP2A2/3.D0
     ,          + DL1**2 * CP2A1/2.D0 + DL1 * CP2A0 + C2DELT
*
       RETURN
       END
*
* =================================================================av==
