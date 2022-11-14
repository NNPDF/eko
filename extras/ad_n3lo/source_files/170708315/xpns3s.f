*
* ..File: xpns3s.f
*   
* ..The 4-loop MSbar splitting functions P_ns^(3)s, which contributes
*    to the N^3LO evolution of the total valence quark distribution. 
*    The expansion parameter is alpha_s/(4 pi), the scale mu_r = mu_f.
*
* ..The nf^2 part is a high-accuracy (0.1% or better) parametrization
*    of the exact expression obtained in arXiv:1610.07477, see xpns3m.f
*
* ..The nf^0 part is an approximation based on the first 9 odd moments.
*    The two sets spanning the error estimate are called via  IMOD = 1
*    and  IMOD = 2.  Any other value of IMOD invokes their average.
*
* ..The distributions (in the mathematical sense) are given as in eq.
*    (B.27) of Floratos, Kounnas, Lacaze: Nucl. Phys. B192 (1981) 417.
*    The name-endings A, B, and C of the functions below correspond to 
*    the kernel superscripts [2], [3], and [1] in that equation.
*
*  ..Reference: 
*        S. Moch, B. Ruijl, T. Ueda, J.Vermaseren and A. Vogt,
*         DESY 17-106, Nikhef 2017-034, LTH 1139
*
*  ..Additional reference for the nf^2 part
*        J. Davies, B. Ruijl, T. Ueda, J.Vermaseren and A. Vogt
*         arXiv:1610.07477, Nucl. Phys. B915 (2017) 335
*
* =====================================================================
*
*
* ..The regular piece of P_ns^(3)s. 
*
       FUNCTION P3NSSA (Y, NF, IMOD)
*
       IMPLICIT REAL*8 (A-Z)
       INTEGER IMOD, nf
*
       Y1  = 1.D0-Y
       DM  = 1.D0/Y1
       DL  = LOG (Y)
       DL1 = LOG (1.D0-Y)
*
* nf^1: two approximations
*
       P3NSA11 = Y1*Y* ( 4989.2 - 1607.73* Y)
     ,         + 3687.6* DL + 3296.6* DL**2 + 1271.11* DL**3
     ,         + 533.44 * DL**4 + 97.27*  DL**5 + 4.* DL**6
     ,         + 60.40* Y1*DL1**2 + 4.685* Y1*DL1**3
       P3NSA12 = 1030.79* Y1*Y + 1266.77* Y1*(2.D0-Y*Y) 
     ,         + 2987.83 * DL + 273.05* DL**2 - 923.48* DL**3 
     ,         - 236.76* DL**4 - 33.886*  DL**5 - 4.* DL**6 
     ,         - 254.63* Y1*DL1 - 0.28953* Y1*DL1**3
*
* nf^2 (parametrized) 
*
       P3NSSA2 = 2.5D+2*  ( Y1* ( -4.7656 + 1.6908* Y + 0.1703* Y**2 ) 
     ,         - 0.41652* Y*DL + 0.90777* Y*DL**2 + 0.12478* Y*DL**3 
     ,         + 0.17155* Y1*DL1 + 0.17191 * DL*DL1 )
     ,         - 6.473971E+2*DL - 6.641219E+1*DL**2 - 5.353347E+0*DL**3
     ,         - 5.925926E+0*DL**4 - 3.950617E-1*DL**5
     ,         + 1.970002E+1*Y1*DL1 - 3.435474E+0*Y1*DL1**2  
*
       IF (IMOD .EQ. 1) THEN
         P3NSSA = nf*P3NSA11 + nf**2* P3NSSA2
       ELSE IF (IMOD .EQ. 2) THEN
         P3NSSA = nf*P3NSA12 + nf**2* P3NSSA2
       ELSE
         P3NSSA = 0.5*nf* (P3NSA11 + P3NSA12) + nf**2* P3NSSA2 
       END IF
*
       RETURN
       END
*
* =================================================================av==
