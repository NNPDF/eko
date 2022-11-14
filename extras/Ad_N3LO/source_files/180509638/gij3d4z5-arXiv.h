* ----------------------------------------------------------------------------
*
*   Four-loop anomalous dimensions gamma_ns^+-(N) and gamma_ij(N), i,j = q,g
*   -- zeta_5 parts of the contributions with quartic Casimir invariants --
*
*   Reference: On quartic colour factors in splitting functions
*              and the gluon cusp anomalous dimension
*
*              S. Moch, B. Ruijl, T. Ueda, J.A.M. Vermaseren and A. Vogt,
*              DESY 18-072, Nikhef 2018-023, LTH 1165
*
*   The non-singlet (ns) expressions were already given in  
*
*              S. Moch, B. Ruijl, T. Ueda, J.A.M. Vermaseren and A. Vogt,
*              arXiv:1707.08315 = JHEP 10 (2017) 041
*
*   Notations: S(R(..),N) = harmonic sums, z5 = zeta_5 (Riemann's zeta-fct.),
*              den(N+a) = 1/(N+a), sign(N) -> 1 for gamma_ns^+ and gamma_ij.
*              Expansion in alpha_s/(4*pi), `-' relative to splitting fct's.
*
*   The diagonal entries are reciproxity respecting (RR); the off-diagonal
*   quantities are closely related and part of RR expressions, see the paper.
*
* ----------------------------------------------------------------------------

 L gns3z5N = 

  + [d4RA/nc]* 320/3*z5* (
     - 75/2 
     + 1/2 * sign(N)
     + 163* (den(N)-den(N+1))
     + 13/2 * (den(N)-den(N+1)) * sign(N)
     + 58 * S(R(1),N)
     - 69 * (den(N)^2 + den(N+1)^2)
     + 24 * S(R(1),N)* (den(N)-den(N+1))
     - 24 * S(R(1),N)^2
     )
  + [d4RR/nc]*nf* 1280/3*z5* (
        3
     - 17 * (den(N)- den(N+1))
     -  2 * S(R(1),N)
     +  6 * (den(N)^2 + den(N+1)^2) 
    );

 L gps3z5N = 

  + [d4RR/nc]*nf* 320/3*z5* (
     - 1
     - 16 * (den(N)-den(N+1))
     - 16 * (den(N-1)-den(N+2))
     + 36 * (den(N)^2 + den(N+1)^2)
    );

 L gqq3z5N = gns3z5N + gps3z5N;    * with sign(N) -> 1 

 L gqg3z5N =
  
  + [d4RA/na]*nf* 640/3*z5* (
       391/2 * den(N)
     - 202 * den(N+1)
     - 30 * den(N+2)
     -  8 * den(N-1) 
     - 63 * den(N)^2
     - 126* den(N+1)^2
     + 24 * den(N+2)^2 
     + 24* (den(N)-2*den(N+1)+2*den(N+2))* S(R(1),N)
     )
  + [d4RR/na]*nf^2* 1280/3*z5* (
     - 68 * den(N)
     + 64 * den(N+1)
     +  8 * den(N+2)
     + 24 * den(N)^2
     + 48 * den(N+1)^2
    );

 L ggq3z5N = 

  + [d4RA/nc]* 320/3*z5* (
     - 202 * den(N)
     + 391/2 * den(N+1)
     - 30 * den(N-1)
     -  8 * den(N+2)
     + 126 * den(N)^2
     + 63 * den(N+1)^2
     - 24 * den(N-1)^2
     + 24 * (2*den(N-1)-2*den(N)+den(N+1))* S(R(1),N)
     )
  + [d4RR/nc]*nf* 640/3*z5* (
       64 * den(N)
     - 68 * den(N+1)
     +  8 * den(N-1)
     - 48 * den(N)^2
     - 24 * den(N+1)^2
    );

 L ggg3z5N = 

  + [d4AA/na]* 64/3*z5* (
     - 751/3
     - 1/6* N*(N+1)
     - 532 * (den(N)-den(N+1))
    + 10* (
     - 13 * (den(N-1)-den(N+2))
     + 33 * S(R(1),N)
     + 36 * (den(N)^2+den(N+1)^2)
     - 12 * (den(N-1)^2+den(N+2)^2)
     - 24 * (den(N)-den(N+1))* S(R(1),N)
     + 24 * (den(N-1)-den(N+2))* S(R(1),N)
     - 12 *S(R(1),N)^2 
     )
    )
  + [d4RA/na]*nf* 128/3*z5* (
       1/3 * N*(N+1)
     + 287/3
     - 421 * (den(N)-den(N+1))
     + 20 * (den(N-1)-den(N+2))
     + 150* (den(N)^2 + den(N+1)^2)
     - 60 * S(R(1),N)
    )
  + [d4RR/na]*nf^2 * 128/3*z5* ( 
     - 1/3 * N*(N+1)
     - 17/3 
     + 526 * (den(N)-den(N+1))
     - 240 * (den(N)^2 + den(N+1)^2)   
    ); 

* ----------------------------------------------------------------------------
