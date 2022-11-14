*
* ..File: xcpqg1e.f    F_phi,Q and F_phi,G
*
*
* ..The 1-loop MS(bar) quark and gluon coefficient functions for the 
*    structure function F_phi in deep-inelastic scattering at mu_f = Q. 
*    The expansion parameter is alpha_s/(4 pi).
* 
* ..The distributions (in the mathematical sense) are given as in eq.
*    (B.26) of Floratos, Kounnas, Lacaze: Nucl. Phys. B192 (1981) 417.
*    The name-endings A, B, and C of the functions below correspond to 
*    the kernel superscripts [2], [3], and [1] in that equation.
*
* ..Reference: G. Soar, S. Moch, J. Vermaseren and A. Vogt,
*              arXiv:0912.0369 [hep-ph] 
*
* =====================================================================
*
*
* ..The quark coefficient function
*
       FUNCTION CPHIQ1A (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       CF  = 4.D0/3.D0
*
       CPHIQ1A = CF * ( - 3.D0/Y + 2.D0*Y
     ,              + (LOG(Y) - LOG(1.D0-Y))* (4.D0 -4.D0/Y - 2.D0*Y) )
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The regular piece of the gluon coefficient function
*
       FUNCTION CPHIG1A (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       CA = 3.D0
       DY = 1./Y
       DL = LOG(Y)
*
       XPG1AA = - 11.D0/3.D0 * DY + (DL - LOG(1.-Y)) * 
     ,          (8.D0 - 4.D0*DY - 4.D0*Y + 4.D0*Y**2) - DL* 4.D0/(1.-Y)
       XPG1AF = 2.D0/3.D0 * DY
       CPHIG1A  = CA * XPG1AA + NF * XPG1AF 
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The singular (soft) piece

       FUNCTION CPHIG1B (Y, NF)
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
*
       CA  = 3.D0
*
       CPHIG1B = ( CA * (4.D0*LOG (1.D0-Y)-11.D0/3.D0) 
     1           + NF * 2.D0/3.D0) / (1.D0-Y)
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
*
* ..The 'local' piece
*
       FUNCTION CPHIG1C (Y, NF)
*
       IMPLICIT REAL*8 (A - Z)
       INTEGER NF
       PARAMETER ( Z2 = 1.6449 34066 84822 64365 D0 )
*
       CA  = 3.D0
*
       C1DELT = CA * ( 67.D0/9.D0 - 4.D0* Z2 ) - NF * 10.D0/9.D0
*
       DL1 = LOG (1.D0-Y)
*
       CPHIG1C = CA* (2.D0*DL1**2-11.D0/3.D0*DL1) 
     1         + NF* 2.D0/3.D0*DL1 + C1DELT
*
       RETURN
       END
*
* =================================================================av==
