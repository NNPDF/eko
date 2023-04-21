(* ::Package:: *)

(*
Non Singlet asymptotic limit, valid both for nsm, nsp, nsv
see 1707.08315, euqations: 2.17, 3.11, 3.10, 3.9
*)

(*
Note: A_n are the QCD cusp factor, see qcd_cusp.m
*)
A1 = QCDcusp1 4^1;
A2 = QCDcusp2 4^2;
A3 = QCDcusp3 4^3;
A4 = QCDcusp4 4^4;

(* B1, coefficient of delta(1-x) in eq 4.5 of 0403192 *)
B1= 3 cf;

(* B2, coefficient of delta(1-x) in eq 4.6 of 0403192 *)
B2 = 4 ca cf (17/24+ 11/3 z2 - 3 z3 ) - 4 cf nf (1/12 + 2/3 z2) + 4 cf^2 (3/8 - 3z2 +6 z3);

(*
B3, coefficient of delta(1-x) in eq 4.9 of 0403192,
this is copied form https://www.nikhef.nl/~avogt/xpns2e.f,
see P2DELT
*)
B3 = (
    + 29/2*cf^3
    + 151/4*ca*cf^2
    - 1657/36*ca^2*cf
    - 240*z5*cf^3
    + 120*z5*ca*cf^2
    + 40*z5*ca^2*cf
    + 68*z3*cf^3
    + 844/3*z3*ca*cf^2
    - 1552/9*z3*ca^2*cf
    + 18*z2*cf^3
    - 410/3*z2*ca*cf^2
    + 4496/27*z2*ca^2*cf
    - 32*z2*z3*cf^3
    + 16*z2*z3*ca*cf^2
    + 288/5*z2^2*cf^3
    - 988/15*z2^2*ca*cf^2
    - 2*z2^2*ca^2*cf
    - 1336/27*z2*ca*cf*nf
    + 4/5*z2^2*ca*cf*nf
    + 200/9*z3*ca*cf*nf
    + 20*ca*cf*nf
    + 20/3*z2*cf^2*nf
    + 232/15*z2^2*cf^2*nf
    - 136/3*z3*cf^2*nf
    - 23*cf^2*nf
    + 80/27*z2*cf*nf^2
    - 16/9*z3*cf*nf^2
    - 17/9*cf*nf^2
);

(* B4, eq 3.9, only the leading color term is provided *)
B4 = (cf nc^3 (
		\[Minus] 1379569 / 5184 + 24211/27 z2 \[Minus] 9803/162 z3 \[Minus] 9382/9 z4 + 838/9 z2 z3 + 1002 z5
		+ 16/3 z3^2 + 135 z6 - 80 z2 z5 + 32 z3 z4 - 560 z7
		)
	+ cf nc^2 nf ( 353/3 \[Minus] 85175/162 z2 \[Minus] 137/9 z3 + 16186/27 z4 \[Minus] 584/9 z2 z3 \[Minus] 248/3 z5 \[Minus] 16/3 z3^2 \[Minus] 144 z6)
	- cf nc nf^2 ( 127/18 \[Minus] 5036/81 z2 + 932/27 z3 + 1292/27 z4 \[Minus] 160/9 z2 z3 \[Minus] 32/3 z5)
	\[Minus] cf nf^3 ( 131/81 \[Minus] 32/81 z2 \[Minus] 304/81 z3 + 32/27 z4)
);

(* eq 3.11 of 1707.08315 *)
C4 = A2^2 + 2 A1 A3;
D4 = A1 (B3 - beta2) + A2 (B2- beta1) + A3(B1 - beta0) /. BetaRules;
