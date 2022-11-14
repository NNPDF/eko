(* ::Package:: *)

(* 
Small-x from 2202.10362. 
This is valid in the large n_c limit so all the
constanst are to be replaced by LargeNcQCDConstantsRules.
With these assumptions NS,+ = NS,- = NS,v.
*)
(* Limit for N \[Rule] 0  *)
(* eq 3.3 *)
Pns3R =(
    + 80*n^(-7)*cf^4
    - 80*n^(-6)*cf^3*beta0
    + 160*n^(-6)*cf^4
    + 24*n^(-5)*cf^2*beta0^2
    + 160*n^(-5)*cf^3*ca
    + 80*n^(-5)*cf^3*beta0
    + 128*n^(-5)*cf^4
    - 480*n^(-5)*z2*cf^2*ca^2
    + 1536*n^(-5)*z2*cf^3*ca
    - 1600*n^(-5)*z2*cf^4
      ) /. BetaRules;
(* eq 3.8 *)
p33p =(
    + 16/27*cf*nf^3
    - 88/9*cf*ca*nf^2
    + 484/9*cf*ca^2*nf
    - 2662/27*cf*ca^3
    - 304/9*cf^2*nf^2
    + 4184/9*cf^2*ca*nf
    - 13060/9*cf^2*ca^2
    + 32*cf^3*nf
    - 20*cf^3*ca
    + 212*cf^4
    - 288*z3*cf^3*ca
    + 640*z3*cf^4
    - 48*z2*cf*nc^2*nf
    + 192*z2*cf*nc^3
      );
(* eq. 3.9 *)
p34p =(
    - 176/81*cf*nf^3
    + 1288/27*cf*ca*nf^2
    - 2780/9*cf*ca^2*nf
    + 50006/81*cf*ca^3
    + 4288/81*cf^2*nf^2
    - 65936/81*cf^2*ca*nf
    + 229480/81*cf^2*ca^2
    - 340/3*cf^3*nf
    + 196*cf^3*ca
    - 224*cf^4
    + 236*z4*cf*nc^3
    - 128*z3*cf^2*ca*nf
    + 64*z3*cf^2*ca^2
    + 512/3*z3*cf^3*nf
    + 832/3*z3*cf^3*ca
    - 512*z3*cf^4
    - 80/9*z2*cf*nc*nf^2
    + 1552/9*z2*cf*nc^2*nf
    - 7682/9*z2*cf*nc^3
	);
(* eq. 3.10*)
p35p =(
    + 64/27*cf*nf^3
    - 7561/81*cf*ca*nf^2
    + 64481/81*cf*ca^2*nf
    - 146482/81*cf*ca^3
    - 7736/81*cf^2*nf^2
    + 90538/81*cf^2*ca*nf
    - 254225/81*cf^2*ca^2
    - 500/3*cf^3*nf
    + 2761/3*cf^3*ca
    + 130*cf^4
    + 240*z5*cf^2*ca^2
    + 960*z5*cf^3*ca
    - 1920*z5*cf^4
    + 328/3*z4*cf*nc^2*nf
    - 1312/3*z4*cf*nc^3
    - 32/3*z3*cf*ca*nf^2
    + 32/3*z3*cf*ca^2*nf
    + 264*z3*cf*ca^3
    + 1328/3*z3*cf^2*ca*nf
    - 8984/3*z3*cf^2*ca^2
    - 2080/3*z3*cf^3*nf
    + 12448/3*z3*cf^3*ca
    + 944*z3*cf^4
    + 272/9*z2*cf*nc*nf^2
    - 4006/9*z2*cf*nc^2*nf
    + 12221/9*z2*cf*nc^3
    - 48*z2*z3*cf*nc^3
);

(* NS,+ *)     
(* Now use eq  2.22 to reconstrict the limit *)
gqq3nspN0asy = -(Pns3R + 1/n^4 p33p + 1/n^3 p34p + 1/n^2 p35p);

(* NS,- ,see paper conclusion *)
gqq3nsmN0asy = gqq3nspN0asy;
