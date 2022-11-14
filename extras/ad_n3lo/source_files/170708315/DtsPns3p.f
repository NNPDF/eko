*
* ..File: DtsPns3p.f  (parametrizations)
*
* ..The differences between the final-state (timelike) & initial-state 
*   (space-like) 4-loop MSbar non-singlet splitting function P_ns^(3)a
*   for a = +,-,s. Expansion in alpha_s/(4 pi), scale mu_r = mu_f.
*
* ..Here the exact epressions have been parametrized in terms of powers
*   and logarithms of x. The small-x & large-x coefficients with D`i'
*   are exact up to their truncation to seven digits. The accuracy of 
*   the nf^`j' coefficients is better than 0.1% except close to zeros.   
*   
* ..References:
*
*       A. Mitov, S. Moch and A. Vogt, 
*       Phys. Lett. B638 (2006), hep-ph/0604053  [eq. (20)]
*
*       S. Moch, B. Ruijl, T. Ueda, J.Vermaseren and A. Vogt,
*       DESY 17-106, Nikhef 2017-034, LTH 1139
*
*
* =====================================================================
*
* ..The plus case, delta_{ts} P_{ns}^{(3)+}.
*
       FUNCTION DtsPns3p (x,nf)
       IMPLICIT REAL*8 (A - Z)
       INTEGER nf
*
       x1 = 1.d0-x
       L0 = log(x)
       L1 = log(x1)
*
       DP3pl0 = 2.5D4* ( x1* ( 1.2960 + 1.7438* x - 1.0943* x**2
     ,        - 0.44064* x**3 ) + x*L0* (0.6440 + 0.8939* L0
     ,        + 0.21405* L0**2) + L1 * (2.0343* x1 + 0.35738* L0) )
     ,        - 1.039974D4*L0 - 2.571824D4*L0**2 - 5.965487D3*L0**3
     ,        - 2.067846D2*L0**4 + 4.213992D0*L0**5 - 7.023320D-1*L0**6
     ,        - 3.239247D4 - 3.390187D4* L1
*
       DP3pl1 = 2.5D+2* ( x1* ( -19.877 - 8.0977* x + 12.335* x**2
     ,        + 8.1174* x**3 ) + x*L0* (13.617 - 7.8856* L0
     ,        - 2.2491* L0**2) + L1 * (-20.171* x1 + 6.571* L0) )
     ,        + 6.575425D2*L0 + 3.102901D3*L0**2 + 7.350891D2*L0**3
     ,        + 4.582716D1*L0**4 + 4.975255D3 + 5.483660D3* L1
*
       DP3pl2 = 5.D1* ( x1* ( 1.6030 + 15.938* x - 5.3145* x**2
     ,        + 1.8682* x**3 ) + x*L0* (13.301 + 2.1060* L0
     ,        + 0.4375* L0**2) + L1 * (13.060* x1 + 11.023* L0) )
     ,        - 9.550559D0*L0 - 5.698805D1*L0**2 - 2.159671D1*L0**3
     ,        - 1.580247D0*L0**4 - 8.032433D1 - 5.337723D1* L1
*
       DtsPns3p = DP3pl0 + nf* (DP3pl1 + nf* DP3pl2)
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
* ..The minus case, delta_{ts} P_{ns}^{(3)-}.
*
       FUNCTION DtsPns3m (x,nf)
       IMPLICIT REAL*8 (A - Z)
       INTEGER nf
*
       x1 = 1.d0-x
       L0 = log(x)
       L1 = log(x1)
*
       DP3mn0 = 2.5D4* ( x1* ( 1.2892 + 1.4892* x - 1.4262* x**2
     ,        + 0.29374* x**3 ) + x*L0* (1.1307 + 0.17484* L0
     ,        + 0.14894* L0**2) + L1 * (5.0547* x1 + 3.4824* L0) )
     ,        - 7.307364D3*L0 - 2.461782D4*L0**2 - 7.051323D3*L0**3
     ,        - 6.650339D2*L0**4 - 1.382716D1*L0**5 + 1.035940D0*L0**6
     ,        - 3.239247D4 - 3.390187D4* L1
*
       DP3mn1 = 2.5D+3* ( x1* ( - 1.9867 - 11.407* x + 3.9156* x**2
     ,        - 1.6032* x**3 ) + x*L0* ( - 11.069 - 1.1039* L0
     ,        - 0.2778* L0**2) + L1 * ( - 13.824* x1 - 11.688* L0) )
     ,        + 3.462303D2*L0 + 2.994194D3*L0**2 + 7.804122D2*L0**3
     ,        + 6.689712D1*L0**4 + 1.580247D0*L0**5
     ,        + 4.975255D3 + 5.483660D3* L1
*
       DP3mn2 = 5.D1* ( x1* ( 1.6030 + 15.938* x - 5.3145* x**2
     ,        + 1.8682* x**3 ) + x*L0* (13.301 + 2.1060* L0
     ,        + 0.4375* L0**2) + L1 * (13.060* x1 + 11.023* L0) )
     ,        - 9.550559D0*L0 - 5.698805D1*L0**2 - 2.159671D1*L0**3
     ,        - 1.580247D0*L0**4 - 8.032433D1 - 5.337723D1* L1
*
       DtsPns3m = DP3mn0 + nf* (DP3mn1 + nf* DP3mn2)
*
       RETURN
       END
*
* ---------------------------------------------------------------------
*
* ..The `singlet' case, delta_{ts} P_{ns}^{(3)s}.
*
       FUNCTION DtsPns3s (x,nf)
       IMPLICIT REAL*8 (A - Z)
       INTEGER nf
*
       x1 = 1.d0-x
       L0 = log(x)
       L1 = log(x1)
*
       DP3sg1 = 5.D2* ( x1*x1* ( 0.0597 - 11.761* x + 3.0470* x**2
     ,        - 0.8633* x**3 ) + x*L0* ( - 22.843* x1 - 10.899* L0
     ,        - 3.1331* L0**2) + x1*L1* (x1 + L0)* 0.3835 )
     ,        + 2.970894D2*x1*L0 - 9.498383D2*L0**2 - 4.317969D2*L0**3
     ,        + 5.221778D1*L0**4 + 7.901235D0*L0**5 - 1.580247D0*L0**6
     ,        + x1*x1*L1* ( 1.903189D2 - 2.748379D1* L1 )
*
       DtsPns3s = nf* DP3sg1 
*
       RETURN
       END
*
* =================================================================av==
