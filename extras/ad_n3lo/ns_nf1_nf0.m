(* ::Package:: *)

(*
Non Singlet nf^0 and nf^1 parts
Expessions are taken from 1707.08315
*)
eta = 1/(n)*1/(n+1);
nu = 1/(n-1)*1/(n+2);
D1 = 1/(n+1);

(* eq 3.6, 3.7 *)
gqq3Lncnf0 = (
	cf*nc^3 * (
        8 * (
          - 64*S[1,1,1,3,1,n]
          + 80*S[1,1,1,4,n]
          + 64*S[1,1,3,1,1,n]
          - 64*S[1,1,3,2,n]
          - 48*S[1,1,4,1,n]
          + 16*S[1,1,5,n]
          - 24*S[1,2,2,2,n]
          - 12*S[1,2,4,n]
          - 40*S[1,3,1,2,n]
          - 40*S[1,3,2,1,n]
          + 44*S[1,3,3,n]
          - 104*S[1,4,1,1,n]
          + 100*S[1,4,2,n]
          + 92*S[1,5,1,n]
          - 40*S[1,6,n]
          - 24*S[2,1,2,2,n]
          - 12*S[2,1,4,n]
          - 24*S[2,2,1,2,n]
          - 24*S[2,2,2,1,n]
          + 24*S[2,2,3,n]
          - 72*S[2,3,1,1,n]
          + 76*S[2,3,2,n]
          + 72*S[2,4,1,n]
          - 32*S[2,5,n]
          - 24*S[3,1,1,2,n]
          - 24*S[3,1,2,1,n]
          + 28*S[3,1,3,n]
          - 24*S[3,2,1,1,n]
          + 48*S[3,2,2,n]
          + 44*S[3,3,1,n]
          - 28*S[3,4,n]
          - 24*S[4,1,1,1,n]
          + 56*S[4,1,2,n]
          + 56*S[4,2,1,n]
          - 52*S[4,3,n]
          + 88*S[5,1,1,n]
          - 72*S[5,2,n]
          - 56*S[6,1,n]
          + 20*S[7,n]
          + 32*S[1,1,3,1,n]*eta
          - 40*S[1,1,4,n]*eta
          - 32*S[1,3,1,1,n]*eta
          + 32*S[1,3,2,n]*eta
          + 24*S[1,4,1,n]*eta
          - 8*S[1,5,n]*eta
          + 12*S[2,2,2,n]*eta
          + 6*S[2,4,n]*eta
          + 20*S[3,1,2,n]*eta
          + 20*S[3,2,1,n]*eta
          - 22*S[3,3,n]*eta
          + 52*S[4,1,1,n]*eta
          - 50*S[4,2,n]*eta
          - 46*S[5,1,n]*eta
          + 20*S[6,n]*eta
          + 16*S[1,1,3,n]*eta^2
          + 12*S[1,2,2,n]*eta^2
          - 24*S[1,2,2,n]*D1^2
          - 12*S[1,3,1,n]*eta^2
          - 32*S[1,3,1,n]*D1^2
          - 2*S[1,4,n]*eta^2
          + 28*S[1,4,n]*D1^2
          + 12*S[2,1,2,n]*eta^2
          - 24*S[2,1,2,n]*D1^2
          + 12*S[2,2,1,n]*eta^2
          - 24*S[2,2,1,n]*D1^2
          - 20*S[2,3,n]*eta^2
          + 24*S[2,3,n]*D1^2
          + 32*S[3,1,1,n]*eta^2
          - 40*S[3,1,1,n]*D1^2
          - 32*S[3,2,n]*eta^2
          + 44*S[3,2,n]*D1^2
          - 22*S[4,1,n]*eta^2
          + 48*S[4,1,n]*D1^2
          + 14*S[5,n]*eta^2
          - 24*S[5,n]*D1^2
          + 12*S[1,1,2,n]*eta^3
          + 12*S[1,2,1,n]*eta^3
          - 22*S[1,3,n]*eta^3
          + 12*S[2,1,1,n]*eta^3
          - 18*S[2,2,n]*eta^3
          - 12*S[2,2,n]*D1^3
          - 8*S[3,1,n]*eta^3
          - 16*S[3,1,n]*D1^3
          + 11*S[4,n]*eta^3
          + 14*S[4,n]*D1^3
          + 12*S[1,1,1,n]*eta^4
          - 24*S[1,1,1,n]*D1^4
          - 17*S[1,2,n]*eta^4
          + 12*S[1,2,n]*D1^4
          - 17*S[2,1,n]*eta^4
          + 12*S[2,1,n]*D1^4
          + 27/2*S[3,n]*eta^4
          - 4*S[3,n]*D1^4
          - 22*S[1,1,n]*eta^5
          - 40*S[1,1,n]*D1^5
          + 29/2*S[2,n]*eta^5
          + 20*S[2,n]*D1^5
          + 7*S[1,n]*eta^6
          - 15*S[1,n]*D1^6
          - 5/4*eta^7
          - 5/2*D1^7
          )
      + 4/3 * (
          - 64*S[1,1,3,1,n]
          + 80*S[1,1,4,n]
          - 88*S[1,2,3,n]
          - 112*S[1,3,1,1,n]
          - 152*S[1,3,2,n]
          - 224*S[1,4,1,n]
          + 280*S[1,5,n]
          - 88*S[2,1,3,n]
          - 156*S[2,2,2,n]
          - 176*S[2,3,1,n]
          + 186*S[2,4,n]
          - 172*S[3,1,2,n]
          - 172*S[3,2,1,n]
          + 286*S[3,3,n]
          - 148*S[4,1,1,n]
          + 342*S[4,2,n]
          + 378*S[5,1,n]
          - 260*S[6,n]
          - 240*S[1,1,3,n]*eta
          + 144*S[1,2,2,n]*eta
          + 416*S[1,3,1,n]*eta
          - 184*S[1,4,n]*eta
          + 144*S[2,1,2,n]*eta
          + 144*S[2,2,1,n]*eta
          + 20*S[2,3,n]*eta
          + 344*S[3,1,1,n]*eta
          - 212*S[3,2,n]*eta
          - 296*S[4,1,n]*eta
          + 4*S[5,n]*eta
          + 216*S[1,1,2,n]*eta^2
          + 216*S[1,2,1,n]*eta^2
          - 158*S[1,3,n]*eta^2
          - 88*S[1,3,n]*D1^2
          + 216*S[2,1,1,n]*eta^2
          - 216*S[2,2,n]*eta^2
          - 228*S[2,2,n]*D1^2
          - 252*S[3,1,n]*eta^2
          - 304*S[3,1,n]*D1^2
          + 103*S[4,n]*eta^2
          + 310*S[4,n]*D1^2
          + 288*S[1,1,1,n]*eta^3
          - 274*S[1,2,n]*eta^3
          - 24*S[1,2,n]*D1^3
          - 274*S[2,1,n]*eta^3
          - 24*S[2,1,n]*D1^3
          + 160*S[3,n]*eta^3
          - 20*S[3,n]*D1^3
          - 436*S[1,1,n]*eta^4
          - 420*S[1,1,n]*D1^4
          + 229*S[2,n]*eta^4
          + 192*S[2,n]*D1^4
          + 108*S[1,n]*eta^5
          - 368*S[1,n]*D1^5
          - 25/2*eta^6
          - 95*D1^6
          )
      +  2/9* (
          + 3216*S[1,2,2,n]
          + 5788*S[1,3,1,n]
          - 4745*S[1,4,n]
          + 3216*S[2,1,2,n]
          + 3216*S[2,2,1,n]
          - 3635*S[2,3,n]
          + 3684*S[3,1,1,n]
          - 5581*S[3,2,n]
          - 6899*S[4,1,n]
          + 4086*S[5,n]
          + 1272*S[1,3,n]*eta
          - 912*S[2,2,n]*eta
          - 3350*S[3,1,n]*eta
          + 3233/2*S[4,n]*eta
          + 864*S[1,1,1,n]*eta^2
          - 1446*S[1,2,n]*eta^2
          + 4416*S[1,2,n]*D1^2
          - 1446*S[2,1,n]*eta^2
          + 4416*S[2,1,n]*D1^2
          + 1051*S[3,n]*eta^2
          - 5099*S[3,n]*D1^2
          - 2070*S[1,1,n]*eta^3
          - 1704*S[1,1,n]*D1^3
          + 3187/2*S[2,n]*eta^3
          + 3432*S[2,n]*D1^3
          + 1789*S[1,n]*eta^4
          - 2985*S[1,n]*D1^4
          - 498*eta^5
          - 1254*D1^5
          )
      + 1/27* (
          + 14240*S[1,3,n]
          + 18058*S[2,2,n]
          + 22200*S[3,1,n]
          - 55291/2*S[4,n]
          - 27576*S[1,2,n]*eta
          - 27576*S[2,1,n]*eta
          + 17492*S[3,n]*eta
          - 31698*S[1,1,n]*eta^2
          - 8064*S[1,1,n]*D1^2
          + 23689*S[2,n]*eta^2
          + 46306*S[2,n]*D1^2
          + 32625*S[1,n]*eta^3
          - 14304*S[1,n]*D1^3
          - 12997/2*eta^4
          - 9883/2*D1^4
          - 88832/3*S[1,2,n]
          - 88832/3*S[2,1,n]
          + 71591/2*S[3,n]
          + 828*S[1,1,n]*eta
          - 18725/3*S[2,n]*eta
          + 19757/6*S[1,n]*eta^2
          - 261590/3*S[1,n]*D1^2
          - 15469/3*eta^3
          - 205282/3*D1^3
          - 24211*S[2,n]
          + 278627/3*S[1,n]*eta
          - 231341/6*eta^2
          - 134154*D1^2
          + 84278/3*S[1,n]
          + 534767/12*eta
          + 1379569/192
          )
       + 32/3*z3 * (
          + 6*S[4,n]
          - 6*S[3,1,n]
          - 6*S[1,3,n]
          + 3*S[3,n]*eta
          - 24*S[1,1,n]*eta^2
          + 12*S[2,n]*eta^2
          + 15*S[1,n]*eta^3
          - 3/4*eta^4
          - 13/2*S[3,n]
          + 60*S[1,1,n]*eta
          - 30*S[2,n]*eta
          - 7/2*S[1,n]*eta^2
          - eta^3
          - 31*S[1,n]*eta
          + 13/16*eta^2
          + 211/9*S[1,n]
          + 839/72*eta
          - 1517/96
          )
       + 80*z5 * (
          - 5*eta^2
          + 8*eta
          - 11/4
          )
        )
);

(* eq 3.7 *)
gqq3Lncnf1 =(
	+ cf*nc^2* (
       16 * (
          - 4*S[1,1,3,n]*eta^2
          + 4*S[1,3,1,n]*eta^2
          + 2*S[2,3,n]*eta^2
          - 2*S[4,1,n]*eta^2
          + 2*S[1,3,n]*eta^3
          - 2*S[3,1,n]*eta^3
          )
       + 8/3 * (
          + 32*S[1,1,3,1,n]
          - 40*S[1,1,4,n]
          + 8*S[1,2,3,n]
          - 16*S[1,3,1,1,n]
          + 40*S[1,3,2,n]
          + 40*S[1,4,1,n]
          - 32*S[1,5,n]
          + 8*S[2,1,3,n]
          + 24*S[2,2,2,n]
          + 16*S[2,3,1,n]
          - 12*S[2,4,n]
          + 32*S[3,1,2,n]
          + 32*S[3,2,1,n]
          - 44*S[3,3,n]
          + 56*S[4,1,1,n]
          - 72*S[4,2,n]
          - 72*S[5,1,n]
          + 40*S[6,n]
          + 24*S[1,1,3,n]*eta
          - 40*S[1,3,1,n]*eta
          + 20*S[1,4,n]*eta
          - 16*S[2,3,n]*eta
          + 8*S[3,1,1,n]*eta
          - 20*S[3,2,n]*eta
          - 8*S[4,1,n]*eta
          + 16*S[5,n]*eta
          - 2*S[1,3,n]*eta^2
          + 8*S[1,3,n]*D1^2
          - 12*S[2,2,n]*eta^2
          + 24*S[2,2,n]*D1^2
          - 12*S[3,1,n]*eta^2
          + 32*S[3,1,n]*D1^2
          + 10*S[4,n]*eta^2
          - 32*S[4,n]*D1^2
          - 16*S[1,2,n]*eta^3
          - 16*S[2,1,n]*eta^3
          + 19*S[3,n]*eta^3
          + 4*S[3,n]*D1^3
          - 28*S[1,1,n]*eta^4
          + 48*S[1,1,n]*D1^4
          + 25*S[2,n]*eta^4
          - 24*S[2,n]*D1^4
          + 18*S[1,n]*eta^5
          + 40*S[1,n]*D1^5
          - 5*eta^6
          + 10*D1^6
          )
      + 4/9 * (
          - 240*S[1,2,2,n]
          - 376*S[1,3,1,n]
          + 362*S[1,4,n]
          - 240*S[2,1,2,n]
          - 240*S[2,2,1,n]
          + 362*S[2,3,n]
          - 240*S[3,1,1,n]
          + 550*S[3,2,n]
          + 686*S[4,1,n]
          - 552*S[5,n]
          - 168*S[1,3,n]*eta
          + 24*S[2,2,n]*eta
          + 236*S[3,1,n]*eta
          - 73*S[4,n]*eta
          - 96*S[1,2,n]*eta^2
          - 336*S[1,2,n]*D1^2
          - 96*S[2,1,n]*eta^2
          - 336*S[2,1,n]*D1^2
          + 74*S[3,n]*eta^2
          + 482*S[3,n]*D1^2
          - 432*S[1,1,n]*eta^3
          + 96*S[1,1,n]*D1^3
          + 229*S[2,n]*eta^3
          - 240*S[2,n]*D1^3
          + 152*S[1,n]*eta^4
          + 474*S[1,n]*D1^4
          - 18*eta^5
          + 228*D1^5
          )
       + 2/27 * (
          - 2420*S[1,3,n]
          - 4084*S[2,2,n]
          - 5748*S[3,1,n]
          + 6525*S[4,n]
          + 2016*S[1,2,n]*eta
          + 2016*S[2,1,n]*eta
          - 452*S[3,n]*eta
          + 486*S[1,1,n]*eta^2
          + 576*S[1,1,n]*D1^2
          + 95*S[2,n]*eta^2
          - 7492*S[2,n]*D1^2
          - 63*S[1,n]*eta^3
          + 1848*S[1,n]*D1^3
          - 732*eta^4
          + 1299*D1^4
          + 13346/3*S[1,2,n]
          + 13346/3*S[2,1,n]
          - 22247/3*S[3,n]
          + 288*S[1,1,n]*eta
          + 4943/3*S[2,n]*eta
          + 11624/3*S[1,n]*eta^2
          + 38018/3*S[1,n]*D1^2
          - 8045/6*eta^3
          + 31378/3*D1^3
          + 85175/12*S[2,n]
          - 39854/3*S[1,n]*eta
          + 13405/3*eta^2
          + 119917/4*D1^2
          - 39883/6*S[1,n]
          - 112979/12*eta
          - 3177/2
          )
       + 32/3*z3 * (
          + 12*S[1,1,n]*eta^2
          - 6*S[2,n]*eta^2
          - 6*S[1,n]*eta^3
          + 6*S[1,2,n]
          + 6*S[2,1,n]
          - 4*S[3,n]
          - 12*S[1,1,n]*eta
          + 3*S[2,n]*eta
          - 4*S[1,n]*eta^2
          + 6*S[1,n]*D1^2
          + eta^3
          + 3*D1^3
          + S[2,n]
          + 4*S[1,n]*eta
          + 9/4*eta^2
          + 4*D1^2
          - 317/8*S[1,n]
          + 249/16*eta
          + 705/32
          )
       + 22*z4 * (
          + 4*S[1,n]
          - 2*eta
          - 3
          )
       + 80*z5 * (
          + 2*eta^2
          + 2*S[1,n]
          - 3*eta
          - 1
          )
        )
);

(* eq D.1 *)
gqq3nspz5N = (
  + cf^4 * 320*z5* (
       111/12
     + 1/6 * Sign[n]
     +  6 * eta
     - 29/6 * eta* Sign[n]
     -  9 * eta^2
     -  7 * eta^2* Sign[n]
     + 14 * S[-2,n]
     )
  + cf^3*ca* 320*z5* (
     - 59/4
     - 1/3 * Sign[n]
     - 12 * eta
     + 20/3 * eta* Sign[n]
     + 18 * eta^2
     + 11 * eta^2* Sign[n]
     - 22 * S[-2,n]
     )
  + cf^2*ca^2* 80*z5* (
       67/4
     +  1 * Sign[n]
     + 67/3 * eta
     -  4 * eta* Sign[n]
     + 58/3 * S[1,n]
     - 58 * eta^2
     - 17 * eta^2* Sign[n]
     +  8 * S[1,n]*eta
     -  8 * S[1,n]^2
     + 34 * S[-2,n]
     )
  + cf*ca^3* 80/3*z5* (
       13/4
     - 5/6 * Sign[n]
     + 40/3 * eta
     - 47/6 * eta* Sign[n]
     - 116/3 * S[1,n]
     + 43 * eta^2
     +  3 * eta^2* Sign[n]
     - 16 * S[1,n]*eta
     + 16 * S[1,n]^2
     -  6 * S[-2,n]
     )
  + (d4RA/nr)* 320*z5* (
     - 25/2
     + 1/6 * Sign[n]
     + 25/3 * eta
     + 13/6 * eta* Sign[n]
     + 58/3 * S[1,n]
     - 23 * eta^2
     +  8 * S[1,n]*eta
     -  8 * S[1,n]^2
     )
  + cf^3*nf* 160*z5* (
       3/2
     + 1 * eta
     - 2 * S[1,n]
     )
  + cf^2*ca*nf * 80/3*z5 * (
     - 3/2
     - 1 * eta
     + 2 * S[1,n]
     )
  + cf*ca^2*nf * 80/9*z5 * (
     - 33/2
     - 25 * eta
     + 26 * S[1,n]
     + 12 * eta^2
     )
  + (d4RR/nr)*nf* 1280/3*z5* (
        3
     -  5 * eta
     -  2 * S[1,n]
     +  6 * eta^2
     )
);

(* eq D.2 *)
gqq3nsmz5N = gqq3nspz5N + (
	+ cf^4 320 z5 (-1/6 + 29/6 eta + 7eta^2)
	+ cf^3 ca  320 z5 (1/3 -20/3 eta -11 eta^2)
	+ cf^2ca^2 z5 (-1 + 4 eta + 17 eta^2)
	+ cf ca^3 80/3 z5 (5/6 + 47/6 eta - 3 eta^2)
	+ (d4RA/nr) 320 z5 (-1/6 -13/6 eta)
);

(**********************************************************)
(* x space expresssions from section 4 *)
(**********************************************************)
(* NS, sea *)
x1 = 1-x;
L1 = Log[1-x];
L0 = Log[x];

(* eq 4.19 *)
pqqnssnf1A =(
	60.40 x1 L1^2 +4.685 x1 L1^3 +x1 x(4989.2\[Minus]1607.73x)
	+3687.6 L0 +3296.6 L0^2 +1271.11 L0^3 +533.44 L0^4 +97.27 L0^5 +4L0^6
);

(* eq 4.20 *)
pqqnssnf1B =(
	\[Minus]254.63 x1 L1 \[Minus]0.28953 x1 L1^3 +1030.79 x1 x +1266.77 x1 (2\[Minus]x^2)
	+ 2987.83 L0 + 273.05 L0^2 \[Minus] 923.48 L0^3 \[Minus] 236.76 L0^4 \[Minus] 33.886 L0^5 \[Minus] 4 L0^6
);

(**********************************************************)
(* Ns,-+ common parts eq 4.11 and nf^3 (exact) *)
x1  = 1-x;
DM  = 1/(1-x);
DL  = Log[x];
DL1 = Log[1-x];

(* Leading large-n_c, nf^0 and nf^1, parametrized, regular piece *)
P3NSA0 = (
2.5*10^4 * ( x1* ( 3.5254 + 8.6935* x - 1.5051* x^2
+ 1.8300* x^3 ) + 11.883* x*DL - 0.09066* x*DL^2
+ 11.410* x1*DL1 + 13.376 * DL*DL1 )
+ 5.167133*10^4*DL + 1.712095*10^4*DL^2 + 2.863226*10^3*DL^3
+ 2.978255*10^2*DL^4 + 1.6*10*DL^5 + 5.* 10^-1*DL^6
- 2.973385*10^4 + 1.906980*10^4*DL1
);
P3NSA1 = (
2.5*10^4* ( x1* ( - 0.74077 + 1.4860* x - 0.23631* x^2
+ 0.31584* x^3 ) + 2.5251* x1*DL1 + 2.5203* DL*DL1
+ 2.2242* x*DL - 0.02460* x*DL^2 + 0.00310* x*DL^3 )
- 9.239374*10^3*DL - 2.917312*10^3*DL^2
- 4.305308*10^2*DL^3 - 3.6*10*DL^4 - 4./3.*DL^5
+ 8.115605*10^3 - 3.079761*10^3*DL1
);

P3NSA3  =(
 - 2.426296 - 8.460488* 10^-1* x
+ ( 5.267490* 10^-1* DM - 3.687243 + 3.160494* x )* DL
- ( 1.316872* (DM+1.* 10^-1) - 1.448560*x )* DL^2
- ( 2.633744* 10^-1*DM - 1.31687*10^-1* (1.+x) )* DL^3
);

(**********************************************************)
(* NS, - : eq. 4.11, 4.15, 4.16, 4.17 *)

(* ..The regular piece of P_ns^(3)-. *)

(* Nonleading large-n_c, nf^0 and nf^1: two approximations *)
P3NMA01 = (
(5992.88* (1.+2.*x) + 31321.44* x*x)*x1 + 511.228
- 1618.07* DL + 2.25480* DL^3 + 31897.82* DL1*x1
+ 4653.76* DL1^2*x1 + 4.964335*10^-1* (DL^6 + 6.*DL^5)
- 2.601749*10^3 - 2.118867*10^3*DL1
);
P3NMA02 = (
( 4043.59 - 15386.6* x)* x*x1 + 502.481
+ 1532.96 * DL^2 + 31.6023* DL^3 - 3997.39 * DL1*x1
+ 511.567* DL1^3*x1 + 4.964335*10^-1* (DL^6 + 18.*DL^5)
- 2.601749*10^3 - 2.118867*10^3*DL1
);
P3NMA11 = (
(114.457* (1.+2.*x) + 2570.73* x*x)*x1 - 7.08645
- 127.012* DL^2 + 2.69618* DL^4 + 1856.63* DL1*x1
+ 440.17* DL1^2*x1 + 3.121643*10^2 + 3.379310*10^2*DL1
);
P3NMA12 = (
(-335.995* (2.+x) -1605.91* x*x)*x1 - 7.82077
- 9.76627* DL^2 + 0.14218* DL^5 - 1360.04* DL1*x1
+ 38.7337* DL1^3*x1 + 3.121643*10^2 + 3.379310*10^2*DL1
);
(* nf^2 (parametrized) *)
P3NSMA2 = (
2.5*10^2*  ( x1* ( 3.2206 + 1.7507* x + 0.13281* x^2
+ 0.45969* x^3 ) + 1.5641* x*DL - 0.37902* x*DL^2
- 0.03248* x*DL^3 + 2.7511* x1*DL1 + 3.2709 * DL*DL1 )
+ 4.378810*10^2*DL + 1.282948*10^2*DL^2 + 1.959945*10*DL^3
+ 9.876543* 10^-1*DL^4 - 3.760092*10^2 + 2.668861*10^1*DL1
);
P3NSMA = (
	P3NSA0 + nf*P3NSA1 + nf^2*P3NSMA2 + nf^3*P3NSA3
	+ 0.5* ((P3NMA01+P3NMA02) + nf* (P3NMA11+P3NMA12))
);
(* ..The singular piece. *)
P3NSMB = (
	(
	2.120902*10^4
	- 5.179372*10^3* nf
	+ 1.955772*10^2* nf^2
	+ 3.272344* nf^3
	)
	+ 0.5* (
		-511.228 + 7.08645*nf
		-502.481 + 7.82077*nf
		)
	)* 1/(1-x);
(* ..The 'local' piece. *)
P3NSMC = ((
	(
	2.120902*10^4
	- 5.179372*10^3* nf
	+ 1.955772*10^2* nf^2
	+ 3.272344* nf^3
	)
	+ 0.5* (
		-511.228 + 7.08645*nf
		-502.481 + 7.82077*nf
	))* DL1
	+ (
	2.579609*10^4 + 0.08
	- ( 5.818637*10^3 + 0.97)* nf
	+ ( 1.938554*10^2 + 0.0037)* nf^2
	+   3.014982* nf^3
	)
	+ 0.5*(
		-2426.05  + 266.674*nf - 0.05*nf
		-2380.255 + 270.518*nf - 0.05*nf
	)
);

(**********************************************************)

(* NS, + : eq. 4.12, 4.13, 4.14 *)

(* ..The regular piece of P_ns^(3)+. *)

(* Nonleading large-n_c, nf^0 and nf^1: two approximations *)
P3NPA01 = (
	3948.16* x1 - 2464.61* (2*x-x*x)*x1
  - 1839.44* DL^2 - 402.156* DL^3
  - 1777.27* DL1^2*x1 - 204.183 * DL1^3*x1 + 507.152
  - 5.587553*10^1*DL^4 - 2.831276*DL^5
  - 1.488340*10^-1*DL^6 - 2.601749*10^3 - 2.118867*10^3*DL1
);
P3NPA02 = (
 (8698.39 - 10490.47*x)* x*x1
  + 1389.73* DL + 189.576* DL^2
  - 173.936* DL1^2*x1 + 223.078* DL1^3*x1 + 505.209
  - 5.587553*10^1*DL^4 - 2.831276*DL^5
  - 1.488340*10^-1*DL^6 - 2.601749*10^3 - 2.118867*10^3*DL1
);
P3NPA11 = (
	(-1116.34 + 1071.24*x)* x*x1
  - 59.3041* DL^2 - 8.4620* DL^3
  - 143.813* DL1*x1 - 18.8803* DL1^3*x1 - 7.33927
  + 4.658436*DL^4 + 2.798354*10^-1*DL^5
  + 3.121643*10^2 + 3.379310*10^2*DL1
);
P3NPA12 = (
	(-690.151 - 656.386* x*x)* x1
  + 133.702* DL^2 + 34.0569* DL^3
  - 745.573* DL1*x1 + 8.61438* DL1^3*x1 - 7.53662
  + 4.658437*DL^4 + 2.798354*10^-1*DL^5
  + 3.121643*10^2 + 3.379310*10^2*DL1
 );
(* nf^2 (parametrized) *)
P3NSPA2 =( 2.5*10^2*  ( x1* ( 3.0008 + 0.8619* x - 0.12411* x^2
+ 0.31595* x^3 ) - 0.37529* x*DL - 0.21684* x*DL^2
- 0.02295* x*DL^3 + 0.03394* x1*DL1 + 0.40431 * DL*DL1 )
+ 3.930056*10^2*DL + 1.125705*10^2*DL^2 + 1.652675*10^1*DL^3
+ 7.901235*10^-1*DL^4 - 3.760092*10^2 + 2.668861*10^1*DL1
);
P3NSPA = (
	P3NSA0 + nf*P3NSA1 + nf^2*P3NSPA2 + nf^3*P3NSA3
	+ 0.5* ((P3NPA01+P3NPA02) + nf* (P3NPA11+P3NPA12))
);

(* ..The singular piece. *)
A4qI  = (
	2.120902*10^4
	- 5.179372*10^3* nf
	+ 1.955772*10^2* nf^2
	+ 3.272344* nf^3
);
A4ap1 = -507.152 + 7.33927*nf;
A4ap2 = -505.209 + 7.53662*nf;
P3NSPB = (A4qI + 0.5* (A4ap1+A4ap2) )* 1/(1-x);
(* ..The 'local' piece. *)
B4qI = (
	2.579609*10^4 + 0.08
	 - ( 5.818637*10^3 + 0.97)* nf
	+ ( 1.938554*10^2 + 0.0037)* nf^2
	+   3.014982* nf^3
);
B4ap1 = -2405.03 + 267.965*nf;
B4ap2 = -2394.47 + 269.028*nf;
P3NSPC = (
	(A4qI + 0.5*(A4ap1+A4ap2))* DL1 + B4qI + 0.5*(B4ap1+B4ap2)
);


(* eq 3.18, to check the expansion, large nc limit *)
P3nsLnc = (
	+ 16/81*L0^3*(1-x)^(-1)*cf*nf^3
	- 212/27*L0^3*(1-x)^(-1)*cf*nc*nf^2
    + 725/9*L0^3*(1-x)^(-1)*cf*nc^2*nf
    - 55291/324*L0^3*(1-x)^(-1)*cf*nc^3
    - 8/81*L0^3*cf*nf^3
    + 92/27*L0^3*cf*nc*nf^2
    - 851/27*L0^3*cf*nc^2*nf
    + 2987/81*L0^3*cf*nc^3
    - 8/81*L0^3*x*cf*nf^3
    + 20/3*L0^3*x*cf*nc*nf^2
    - 2257/27*L0^3*x*cf*nc^2*nf
    + 38641/162*L0^3*x*cf*nc^3
    + 32/3*L0^3*z3*(1-x)^(-1)*cf*nc^3
    - 28/3*L0^3*z3*cf*nc^3
    - 28/3*L0^3*z3*x*cf*nc^3
    - 32*L0^3*z2*(1-x)^(-1)*cf*nc^2*nf
    + 84*L0^3*z2*(1-x)^(-1)*cf*nc^3
    + 24*L0^3*z2*cf*nc^2*nf
    - 52*L0^3*z2*cf*nc^3
    + 24*L0^3*z2*x*cf*nc^2*nf
    - 340/3*L0^3*z2*x*cf*nc^3
    - 8/9*L0^4*(1-x)^(-1)*cf*nc*nf^2
    + 92/9*L0^4*(1-x)^(-1)*cf*nc^2*nf
    - 227/6*L0^4*(1-x)^(-1)*cf*nc^3
    + 2/3*L0^4*cf*nc*nf^2
    - 65/9*L0^4*cf*nc^2*nf
    + 463/18*L0^4*cf*nc^3
    + 2/3*L0^4*x*cf*nc*nf^2
    - 31/3*L0^4*x*cf*nc^2*nf
    + 251/6*L0^4*x*cf*nc^3
    + 56/3*L0^4*z2*(1-x)^(-1)*cf*nc^3
    - 49/3*L0^4*z2*cf*nc^3
    - 49/3*L0^4*z2*x*cf*nc^3
    + 8/9*L0^5*(1-x)^(-1)*cf*nc^2*nf
    - 26/9*L0^5*(1-x)^(-1)*cf*nc^3
    - 7/9*L0^5*cf*nc^2*nf
    + 22/9*L0^5*cf*nc^3
    - 7/9*L0^5*x*cf*nc^2*nf
    + 34/9*L0^5*x*cf*nc^3
    - 2/9*L0^6*(1-x)^(-1)*cf*nc^3
    + 5/24*L0^6*cf*nc^3
    + 5/24*L0^6*x*cf*nc^3
    - 16/3*H[0,0,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    + 4/3*H[0,0,1,x]*L0^3*cf*nc^3
    + 4/3*H[0,0,1,x]*L0^3*x*cf*nc^3
    - 80/9*H[0,1,x]*L0^3*(1-x)^(-1)*cf*nc^2*nf
    + 188/9*H[0,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    + 56/9*H[0,1,x]*L0^3*cf*nc^2*nf
    - 50/9*H[0,1,x]*L0^3*cf*nc^3
    + 56/9*H[0,1,x]*L0^3*x*cf*nc^2*nf
    - 218/9*H[0,1,x]*L0^3*x*cf*nc^3
    + 8/3*H[0,1,x]*L0^4*(1-x)^(-1)*cf*nc^3
    - 2/3*H[0,1,x]*L0^4*cf*nc^3
    - 2/3*H[0,1,x]*L0^4*x*cf*nc^3
    + 112/3*H[0,1,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    - 88/3*H[0,1,1,x]*L0^3*cf*nc^3
    - 88/3*H[0,1,1,x]*L0^3*x*cf*nc^3
    + 16/27*H[1,x]*L0^3*(1-x)^(-1)*cf*nc*nf^2
    + 724/27*H[1,x]*L0^3*(1-x)^(-1)*cf*nc^2*nf
    - 4745/27*H[1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    - 8/27*H[1,x]*L0^3*cf*nc*nf^2
    - 338/27*H[1,x]*L0^3*cf*nc^2*nf
    + 3977/54*H[1,x]*L0^3*cf*nc^3
    - 8/27*H[1,x]*L0^3*x*cf*nc*nf^2
    - 386/27*H[1,x]*L0^3*x*cf*nc^2*nf
    + 5513/54*H[1,x]*L0^3*x*cf*nc^3
    + 368/3*H[1,x]*L0^3*z2*(1-x)^(-1)*cf*nc^3
    - 184/3*H[1,x]*L0^3*z2*cf*nc^3
    - 184/3*H[1,x]*L0^3*z2*x*cf*nc^3
    + 32/9*H[1,x]*L0^4*(1-x)^(-1)*cf*nc^2*nf
    - 140/9*H[1,x]*L0^4*(1-x)^(-1)*cf*nc^3
    - 16/9*H[1,x]*L0^4*cf*nc^2*nf
    + 58/9*H[1,x]*L0^4*cf*nc^3
    - 16/9*H[1,x]*L0^4*x*cf*nc^2*nf
    + 82/9*H[1,x]*L0^4*x*cf*nc^3
    - 8/3*H[1,x]*L0^5*(1-x)^(-1)*cf*nc^3
    + 4/3*H[1,x]*L0^5*cf*nc^3
    + 4/3*H[1,x]*L0^5*x*cf*nc^3
    + 112/3*H[1,0,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    - 56/3*H[1,0,1,x]*L0^3*cf*nc^3
    - 56/3*H[1,0,1,x]*L0^3*x*cf*nc^3
    - 160/9*H[1,1,x]*L0^3*(1-x)^(-1)*cf*nc^2*nf
    + 160/9*H[1,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    + 80/9*H[1,1,x]*L0^3*cf*nc^2*nf
    + 160/9*H[1,1,x]*L0^3*cf*nc^3
    + 80/9*H[1,1,x]*L0^3*x*cf*nc^2*nf
    - 320/9*H[1,1,x]*L0^3*x*cf*nc^3
    - 16/3*H[1,1,x]*L0^4*(1-x)^(-1)*cf*nc^3
    + 8/3*H[1,1,x]*L0^4*cf*nc^3
    + 8/3*H[1,1,x]*L0^4*x*cf*nc^3
    + 320/3*H[1,1,1,x]*L0^3*(1-x)^(-1)*cf*nc^3
    - 160/3*H[1,1,1,x]*L0^3*cf*nc^3
    - 160/3*H[1,1,1,x]*L0^3*x*cf*nc^3
) /. L0 -> Log[x];
