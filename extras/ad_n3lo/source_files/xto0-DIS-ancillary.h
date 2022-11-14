* -----------------------------------------------------------------------------
*
*   High-energy double logarithms for splitting and coefficient functions in 
*   deep-inelastic scattering
*
*   Reference: Resummation of small-x double logarithms in QCD: 
*              inclusive deep-inelastic scattering
*
*              J. Davies, C.-H. Kom, S. Moch and A. Vogt
*              arXiv:2202.02xxx = LTH 1289, DESY 21-225
*
*   Notation:  z2, z3, z4, z5, etc, are values of Riemann's zeta-function,
*              nf = number of flavours; 
*              SU(nc) colour group invariants are ca = nc, cf.
*
*	           QCD beta-function coefficient beta0 = 11/3*ca - 2/3*nf
*
*              Perturbative expansion in as = alpha_s/(4*pi) 
*
*              N-space functions: S = sqrt(1-4*xi), see eq.(3.1);
*                                 F = 1/sqrt(S), see eq.(5.23).
*
*              x-space logarithms lix=ln(1/x);
*              tI0, tI1, tI2, ... modified Bessel functions, see eq.(3.7);
*              L0 = ln(x); H(R(...),x) are harmonic polylogarithms
*
* -----------------------------------------------------------------------------
*
* Section 3
*
* -----------------------------------------------------------------------------

* eq.(3.2)
L   PnsR =
       - 1/2*S^(-1)*as*beta0
       + S^(-1)*as*cf
       + 1/2*as*beta0
       - as*cf
       + 1/32*N*S^(-3)*as*cf^(-1)*beta0^2
       - 1/8*N*S^(-3)*as*beta0
       + 1/8*N*S^(-3)*as*cf
       - 1/16*N*S^(-1)*as*cf^(-1)*beta0^2
       + 5/6*N*S^(-1)*as*ca
       + 7/6*N*S^(-1)*as*beta0
       + 1/4*N*S^(-1)*as*cf
       - 15/4*N*S^(-1)*as*z2*cf^(-1)*ca^2
       + 12*N*S^(-1)*as*z2*ca
       - 12*N*S^(-1)*as*z2*cf
       + 1/2*N
       - 2*N*as*cf
       + 15/2*N*as*z2*cf^(-1)*ca^2
       - 24*N*as*z2*ca
       + 22*N*as*z2*cf
       - 1/2*N*S
       + 1/32*N*S*as*cf^(-1)*beta0^2
       - 5/6*N*S*as*ca
       - 25/24*N*S*as*beta0
       + 13/8*N*S*as*cf
       - 15/4*N*S*as*z2*cf^(-1)*ca^2
       + 12*N*S*as*z2*ca
       - 10*N*S*as*z2*cf
      ;

* eq.(3.3)
L   Pns3R =
       + 80*N^(-7)*cf^4
       - 80*N^(-6)*cf^3*beta0
       + 160*N^(-6)*cf^4
       + 24*N^(-5)*cf^2*beta0^2
       + 160*N^(-5)*cf^3*ca
       + 80*N^(-5)*cf^3*beta0
       + 128*N^(-5)*cf^4
       - 480*N^(-5)*z2*cf^2*ca^2
       + 1536*N^(-5)*z2*cf^3*ca
       - 1600*N^(-5)*z2*cf^4
      ;

* eq.(3.4)
L   Pns4R =
       + 448*N^(-9)*cf^5
       - 560*N^(-8)*cf^4*beta0
       + 1120*N^(-8)*cf^5
       + 240*N^(-7)*cf^3*beta0^2
       + 3200/3*N^(-7)*cf^4*ca
       + 640/3*N^(-7)*cf^4*beta0
       + 1280*N^(-7)*cf^5
       - 3600*N^(-7)*z2*cf^3*ca^2
       + 11520*N^(-7)*z2*cf^4*ca
       - 11840*N^(-7)*z2*cf^5
      ;

* eq.(3.5)
L   PnsxNNR =
       + 16*tI2*as^3*cf^3*lix^2
       - 60*tI2*as^3*z2*cf*ca^2*lix^2
       + 192*tI2*as^3*z2*cf^2*ca*lix^2
       - 176*tI2*as^3*z2*cf^3*lix^2
       + 2*tI1*as*cf
       - 2*tI1*as^2*cf*beta0*lix
       + 4*tI1*as^2*cf^2*lix
       + tI1*as^3*cf*beta0^2*lix^2
       - 4*tI1*as^3*cf^2*beta0*lix^2
       + 4*tI1*as^3*cf^3*lix^2
       + 20/3*tI0*as^2*cf*ca
       + 22/3*tI0*as^2*cf*beta0
       - 4*tI0*as^2*cf^2
       - 8*tI0*as^2*z2*cf^2
      ;

* eq.(3.8)
L   p33p =
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
      ;

* eq.(3.9)
L   p34p =
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
      ;

* eq.(3.10)
L   p35p =
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
      ;

* eq.(3.11)
L   p43p =
       + 320/27*cf^2*nf^3
       - 1760/9*cf^2*ca*nf^2
       + 9680/9*cf^2*ca^2*nf
       - 53240/27*cf^2*ca^3
       - 2560/9*cf^3*nf^2
       + 35360/9*cf^3*ca*nf
       - 112000/9*cf^3*ca^2
       + 1280/9*cf^4*nf
       + 7120/9*cf^4*ca
       + 1840*cf^5
       - 1920*z3*cf^4*ca
       + 4480*z3*cf^5
       - 240*z2*cf*nc^3*nf
       + 960*z2*cf*nc^4
      ;

* eq.(3.12)
L   p44p =
       + 32/81*cf*nf^4
       - 704/81*cf*ca*nf^3
       + 1936/27*cf*ca^2*nf^2
       - 21296/81*cf*ca^3*nf
       + 29282/81*cf*ca^4
       - 1280/27*cf^2*nf^3
       + 960*cf^2*ca*nf^2
       - 54304/9*cf^2*ca^2*nf
       + 324896/27*cf^2*ca^3
       + 11776/27*cf^3*nf^2
       - 164816/27*cf^3*ca*nf
       + 559624/27*cf^3*ca^2
       - 4376/9*cf^4*nf
       + 9008/9*cf^4*ca
       - 656*cf^5
       + 1240*z4*cf*nc^4
       - 1152*z3*cf^3*ca*nf
       + 2496*z3*cf^3*ca^2
       + 5888/3*z3*cf^4*nf
       - 13376/3*z3*cf^4*ca
       - 512*z3*cf^5
       - 256/3*z2*cf*nc^2*nf^2
       + 10544/9*z2*cf*nc^3*nf
       - 43478/9*z2*cf*nc^4
      ;

* eq.(3.13)
L   p45p =
       - 352/243*cf*nf^4
       + 10384/243*cf*ca*nf^3
       - 34304/81*cf*ca^2*nf^2
       + 423940/243*cf*ca^3*nf
       - 624118/243*cf*ca^4
       + 7424/81*cf^2*nf^3
       - 59554/27*cf^2*ca*nf^2
       + 142534/9*cf^2*ca^2*nf
       - 2795072/81*cf^2*ca^3
       - 7516/9*cf^3*nf^2
       + 79970/9*cf^3*ca*nf
       - 204556/9*cf^3*ca^2
       - 16676/9*cf^4*nf
       + 85946/9*cf^4*ca
       + 500*cf^5
       + 1440*z5*cf^3*ca^2
       + 6400*z5*cf^4*ca
       - 12800*z5*cf^5
       + 984*z4*cf*nc^3*nf
       - 3948*z4*cf*nc^4
       - 192*z3*cf^2*ca*nf^2
       + 832*z3*cf^2*ca^2*nf
       + 1232*z3*cf^2*ca^3
       + 512/3*z3*cf^3*nf^2
       + 7232/3*z3*cf^3*ca*nf
       - 73984/3*z3*cf^3*ca^2
       - 56128/9*z3*cf^4*nf
       + 379168/9*z3*cf^4*ca
       + 5280*z3*cf^5
       - 224/27*z2*cf*nc*nf^3
       + 3584/9*z2*cf*nc^2*nf^2
       - 39236/9*z2*cf*nc^3*nf
       + 321290/27*z2*cf*nc^4
       - 256*z2*z3*cf*nc^4
      ;

* eq.(3.14)
L   p46p =
       + 128/81*cf*nf^4
       - 12826/81*cf*nc*nf^3
       + 760669/243*cf*nc^2*nf^2
       - 5138330/243*cf*nc^3*nf
       + 83997239/1944*cf*nc^4
       - 248*z6*cf*nc^3*nf
       - 1444*z6*cf*nc^4
       + 1504/3*z5*cf*nc^3*nf
       - 4312/3*z5*cf*nc^4
       + 1408/9*z4*cf*nc^2*nf^2
       - 28432/9*z4*cf*nc^3*nf
       + 48070/3*z4*cf*nc^4
       - 64/9*z3*cf*nc*nf^3
       + 2000/9*z3*cf*nc^2*nf^2
       - 4336/3*z3*cf*nc^3*nf
       + 13220/9*z3*cf*nc^4
       - 64*z3^2*cf*nc^3*nf
       + 176*z3^2*cf*nc^4
       + 2656/81*z2*cf*nc*nf^3
       - 9076/9*z2*cf*nc^2*nf^2
       + 267860/27*z2*cf*nc^3*nf
       - 2253859/81*z2*cf*nc^4
       + 128*z2*z3*cf*nc^3*nf
       + 64*z2*z3*cf*nc^4
      ;

* eq.(3.15)
L   p47p =
       - 128/243*cf*nf^4
       + 41497/243*cf*nc*nf^3
       - 1032713/243*cf*nc^2*nf^2
       + 2035745/72*cf*nc^3*nf
       - 141282997/2592*cf*nc^4
       - 112*z7*cf*nc^3*nf
       - 1288*z7*cf*nc^4
       - 248/3*z6*cf*nc^2*nf^2
       + 368*z6*cf*nc^3*nf
       + 1874/3*z6*cf*nc^4
       + 1072/9*z5*cf*nc^2*nf^2
       - 13784/9*z5*cf*nc^3*nf
       + 64174/9*z5*cf*nc^4
       - 176/27*z4*cf*nc*nf^3
       - 4744/9*z4*cf*nc^2*nf^2
       + 71642/9*z4*cf*nc^3*nf
       - 655423/27*z4*cf*nc^4
       + 64/81*z3*cf*nf^4
       + 3008/81*z3*cf*nc*nf^3
       - 18992/27*z3*cf*nc^2*nf^2
       + 197588/81*z3*cf*nc^3*nf
       - 125756/81*z3*cf*nc^4
       + 32*z3*z4*cf*nc^3*nf
       + 1104*z3*z4*cf*nc^4
       - 64/3*z3^2*cf*nc^2*nf^2
       + 944/3*z3^2*cf*nc^3*nf
       - 2056/3*z3^2*cf*nc^4
       - 3376/81*z2*cf*nc*nf^3
       + 47984/27*z2*cf*nc^2*nf^2
       - 141241/9*z2*cf*nc^3*nf
       + 12219019/324*z2*cf*nc^4
       - 64*z2*z5*cf*nc^3*nf
       - 576*z2*z5*cf*nc^4
       + 96*z2*z3*cf*nc^2*nf^2
       - 512/3*z2*z3*cf*nc^3*nf
       - 5488/3*z2*z3*cf*nc^4
      ;

* eq.(3.16)
* put ddd = 1 below; ddd tags terms (...) omitted in eq.(3.16)
L   PnsLxN7 =
       - 2*tI7*as^7*lix^7*nc^7
       + 4*tI7*as^7*z7*lix^7*nc^7
       + 2*tI6*as^6*lix^6*nc^6
       - 4*tI6*as^6*z6*lix^6*nc^6
       - 2*tI5*as^5*lix^5*nc^5
       + 4*tI5*as^5*z5*lix^5*nc^5
       + 46/3*tI5*ddd*as^6*lix^5*nc^6
       + 8/3*tI5*ddd*as^6*beta0*lix^5*nc^5
       + 10*tI5*ddd*as^6*z6*lix^5*nc^6
       - 20*tI5*ddd*as^6*z6*beta0*lix^5*nc^5
       - 44/3*tI5*ddd*as^6*z5*lix^5*nc^6
       + 20/3*tI5*ddd*as^6*z5*beta0*lix^5*nc^5
       - 20*tI5*ddd*as^6*z4*lix^5*nc^6
       - 20*tI5*ddd*as^6*z3*lix^5*nc^6
       + 40*tI5*ddd*as^6*z3*z4*lix^5*nc^6
       - 16*tI5*ddd*as^6*z2*lix^5*nc^6
       + 32*tI5*ddd*as^6*z2*z5*lix^5*nc^6
       + 2*tI4*as^4*lix^4*nc^4
       - 4*tI4*as^4*z4*lix^4*nc^4
       - 22/3*tI4*ddd*as^5*lix^4*nc^5
       - 2/3*tI4*ddd*as^5*beta0*lix^4*nc^4
       - 42*tI4*ddd*as^5*z6*lix^4*nc^5
       - 8*tI4*ddd*as^5*z5*lix^4*nc^5
       + 16*tI4*ddd*as^5*z5*beta0*lix^4*nc^4
       + 32/3*tI4*ddd*as^5*z4*lix^4*nc^5
       - 20/3*tI4*ddd*as^5*z4*beta0*lix^4*nc^4
       + 16*tI4*ddd*as^5*z3*lix^4*nc^5
       - 16*tI4*ddd*as^5*z3^2*lix^4*nc^5
       + 12*tI4*ddd*as^5*z2*lix^4*nc^5
       - 2*tI3*as^3*lix^3*nc^3
       + 4*tI3*as^3*z3*lix^3*nc^3
       + 4/3*tI3*as^4*lix^3*nc^4
       - 4/3*tI3*as^4*beta0*lix^3*nc^3
       + 6*tI3*as^4*z4*lix^3*nc^4
       - 12*tI3*as^4*z4*beta0*lix^3*nc^3
       - 20/3*tI3*as^4*z3*lix^3*nc^4
       + 20/3*tI3*as^4*z3*beta0*lix^3*nc^3
       - 8*tI3*as^4*z2*lix^3*nc^4
       + 16*tI3*as^4*z2*z3*lix^3*nc^4
       + 2737/36*tI3*ddd*as^5*lix^3*nc^5
       - 607/36*tI3*ddd*as^5*beta0*lix^3*nc^4
       + 2*tI3*ddd*as^5*beta0^2*lix^3*nc^3
       - 84*tI3*ddd*as^5*z7*lix^3*nc^5
       + 52*tI3*ddd*as^5*z6*lix^3*nc^5
       - 94*tI3*ddd*as^5*z6*beta0*lix^3*nc^4
       + 31*tI3*ddd*as^5*z5*lix^3*nc^5
       - 88*tI3*ddd*as^5*z5*beta0*lix^3*nc^4
       + 24*tI3*ddd*as^5*z5*beta0^2*lix^3*nc^3
       + 173/2*tI3*ddd*as^5*z4*lix^3*nc^5
       - 81*tI3*ddd*as^5*z4*beta0*lix^3*nc^4
       - 20*tI3*ddd*as^5*z4*beta0^2*lix^3*nc^3
       - 2101/18*tI3*ddd*as^5*z3*lix^3*nc^5
       + 1147/18*tI3*ddd*as^5*z3*beta0*lix^3*nc^4
       - 4/3*tI3*ddd*as^5*z3*beta0^2*lix^3*nc^3
       + 96*tI3*ddd*as^5*z3*z4*lix^3*nc^5
       - 64*tI3*ddd*as^5*z3^2*lix^3*nc^5
       - 16*tI3*ddd*as^5*z3^2*beta0*lix^3*nc^4
       + 2/3*tI3*ddd*as^5*z2*lix^3*nc^5
       + 28/3*tI3*ddd*as^5*z2*beta0*lix^3*nc^4
       + 32*tI3*ddd*as^5*z2*z5*lix^3*nc^5
       - 64/3*tI3*ddd*as^5*z2*z3*lix^3*nc^5
       + 160/3*tI3*ddd*as^5*z2*z3*beta0*lix^3*nc^4
       + 8/3*tI2*as^3*lix^2*nc^3
       + 10/3*tI2*as^3*beta0*lix^2*nc^2
       - 4*tI2*as^3*z3*lix^2*nc^3
       + 8*tI2*as^3*z3*beta0*lix^2*nc^2
       - 4/3*tI2*as^3*z2*lix^2*nc^3
       - 20/3*tI2*as^3*z2*beta0*lix^2*nc^2
       - 3157/36*tI2*ddd*as^4*lix^2*nc^4
       + 1771/36*tI2*ddd*as^4*beta0*lix^2*nc^3
       - 2/3*tI2*ddd*as^4*beta0^2*lix^2*nc^2
       - 82*tI2*ddd*as^4*z6*lix^2*nc^4
       + 12*tI2*ddd*as^4*z5*lix^2*nc^4
       - 12*tI2*ddd*as^4*z5*beta0*lix^2*nc^3
       + 80/3*tI2*ddd*as^4*z4*lix^2*nc^4
       + 148/3*tI2*ddd*as^4*z4*beta0*lix^2*nc^3
       - 12*tI2*ddd*as^4*z4*beta0^2*lix^2*nc^2
       - 377/3*tI2*ddd*as^4*z3*lix^2*nc^4
       + 214/3*tI2*ddd*as^4*z3*beta0*lix^2*nc^3
       + 40/3*tI2*ddd*as^4*z3*beta0^2*lix^2*nc^2
       - 8*tI2*ddd*as^4*z3^2*lix^2*nc^4
       + 2437/18*tI2*ddd*as^4*z2*lix^2*nc^4
       - 1555/18*tI2*ddd*as^4*z2*beta0*lix^2*nc^3
       + 4/3*tI2*ddd*as^4*z2*beta0^2*lix^2*nc^2
       + 72*tI2*ddd*as^4*z2*z3*lix^2*nc^4
       - 8*tI2*ddd*as^4*z2*z3*beta0*lix^2*nc^3
       + tI1
       - 2*tI1*as*nc
       + tI1*as*lix*nc
       - tI1*as*beta0*lix
       + 4*tI1*as*z2*nc
       + 115/12*tI1*as^2*lix*nc^2
       + 1/2*tI1*as^2*lix^2*nc^2
       - 34/3*tI1*as^2*beta0*lix*nc
       - tI1*as^2*beta0*lix^2*nc
       + 1/2*tI1*as^2*beta0^2*lix^2
       + 4507/36*tI1*as^3*lix*nc^3
       + 647/36*tI1*as^3*lix^2*nc^3
       + 1/6*tI1*as^3*lix^3*nc^3
       - 2725/36*tI1*as^3*beta0*lix*nc^2
       - 181/36*tI1*as^3*beta0*lix^2*nc^2
       - 1/2*tI1*as^3*beta0*lix^3*nc^2
       - 139/12*tI1*as^3*beta0^2*lix*nc
       + 343/18*tI1*as^3*beta0^2*lix^2*nc
       + 1/2*tI1*as^3*beta0^2*lix^3*nc
       - 1/6*tI1*as^3*beta0^3*lix^3
       + 60*tI1*as^3*z5*lix*nc^3
       + 12*tI1*as^3*z4*lix*nc^3
       + 45*tI1*as^3*z4*lix^2*nc^3
       - 12*tI1*as^3*z4*beta0*lix*nc^2
       - 103/3*tI1*as^3*z3*lix*nc^3
       + 4*tI1*as^3*z3*lix^2*nc^3
       - 14/3*tI1*as^3*z3*beta0*lix*nc^2
       - 4*tI1*as^3*z3*beta0*lix^2*nc^2
       - 28*tI1*as^3*z2*lix^2*nc^3
       + 8*tI1*as^3*z2*beta0*lix*nc^2
       - 18*tI1*as^3*z2*beta0*lix^2*nc^2
       - 2*tI1*as^3*z2*beta0^2*lix^2*nc
       + 997/72*tI1*as^4*lix^3*nc^4
       + 1/24*tI1*as^4*lix^4*nc^4
       - 35/4*tI1*as^4*beta0*lix^3*nc^3
       - 1/6*tI1*as^4*beta0*lix^4*nc^3
       + 191/24*tI1*as^4*beta0^2*lix^3*nc^2
       + 1/4*tI1*as^4*beta0^2*lix^4*nc^2
       - 235/18*tI1*as^4*beta0^3*lix^3*nc
       - 1/6*tI1*as^4*beta0^3*lix^4*nc
       + 1/24*tI1*as^4*beta0^4*lix^4
       + 45*tI1*as^4*z4*lix^3*nc^4
       - 45*tI1*as^4*z4*beta0*lix^3*nc^3
       + 2*tI1*as^4*z3*lix^3*nc^4
       - 4*tI1*as^4*z3*beta0*lix^3*nc^3
       + 2*tI1*as^4*z3*beta0^2*lix^3*nc^2
       - 82/3*tI1*as^4*z2*lix^3*nc^4
       + 8*tI1*as^4*z2*beta0*lix^3*nc^3
       + 18*tI1*as^4*z2*beta0^2*lix^3*nc^2
       + 4/3*tI1*as^4*z2*beta0^3*lix^3*nc
       + 1/120*tI1*as^5*lix^5*nc^5
       - 1/24*tI1*as^5*beta0*lix^5*nc^4
       + 1/12*tI1*as^5*beta0^2*lix^5*nc^3
       - 1/12*tI1*as^5*beta0^3*lix^5*nc^2
       + 1/24*tI1*as^5*beta0^4*lix^5*nc
       - 1/120*tI1*as^5*beta0^5*lix^5
       + 1/720*tI1*as^6*lix^6*nc^6
       - 1/120*tI1*as^6*beta0*lix^6*nc^5
       + 1/48*tI1*as^6*beta0^2*lix^6*nc^4
       - 1/36*tI1*as^6*beta0^3*lix^6*nc^3
       + 1/48*tI1*as^6*beta0^4*lix^6*nc^2
       - 1/120*tI1*as^6*beta0^5*lix^6*nc
       + 1/720*tI1*as^6*beta0^6*lix^6
       + 1/5040*tI1*as^7*lix^7*nc^7
       - 1/720*tI1*as^7*beta0*lix^7*nc^6
       + 1/240*tI1*as^7*beta0^2*lix^7*nc^5
       - 1/144*tI1*as^7*beta0^3*lix^7*nc^4
       + 1/144*tI1*as^7*beta0^4*lix^7*nc^3
       - 1/240*tI1*as^7*beta0^5*lix^7*nc^2
       + 1/720*tI1*as^7*beta0^6*lix^7*nc
       - 1/5040*tI1*as^7*beta0^7*lix^7
       + 1659497/1728*tI1*ddd*as^4*lix*nc^4
       - 204787/2592*tI1*ddd*as^4*lix^2*nc^4
       - 1147/144*tI1*ddd*as^4*beta0*lix*nc^3
       - 5351/18*tI1*ddd*as^4*beta0*lix^2*nc^3
       - 75427/216*tI1*ddd*as^4*beta0^2*lix*nc^2
       + 81421/216*tI1*ddd*as^4*beta0^2*lix^2*nc^2
       + 523/72*tI1*ddd*as^4*beta0^3*lix*nc
       + 6599/324*tI1*ddd*as^4*beta0^3*lix^2*nc
       - 952*tI1*ddd*as^4*z7*lix*nc^4
       + 84*tI1*ddd*as^4*z7*beta0*lix*nc^3
       + 954*tI1*ddd*as^4*z6*lix*nc^4
       - 525/2*tI1*ddd*as^4*z6*lix^2*nc^4
       - 630*tI1*ddd*as^4*z6*beta0*lix*nc^3
       + 1805/3*tI1*ddd*as^4*z5*lix*nc^4
       + 64*tI1*ddd*as^4*z5*lix^2*nc^4
       + 1000/3*tI1*ddd*as^4*z5*beta0*lix*nc^3
       - 64*tI1*ddd*as^4*z5*beta0*lix^2*nc^3
       + 8*tI1*ddd*as^4*z5*beta0^2*lix*nc^2
       - 23/2*tI1*ddd*as^4*z4*lix*nc^4
       + 2555/6*tI1*ddd*as^4*z4*lix^2*nc^4
       - 207/2*tI1*ddd*as^4*z4*beta0*lix*nc^3
       + 947/3*tI1*ddd*as^4*z4*beta0*lix^2*nc^3
       - 37*tI1*ddd*as^4*z4*beta0^2*lix*nc^2
       + 36*tI1*ddd*as^4*z4*beta0^2*lix^2*nc^2
       - 10117/18*tI1*ddd*as^4*z3*lix*nc^4
       - 244/3*tI1*ddd*as^4*z3*lix^2*nc^4
       + 950/3*tI1*ddd*as^4*z3*beta0*lix*nc^3
       - 113/3*tI1*ddd*as^4*z3*beta0*lix^2*nc^3
       - 460/9*tI1*ddd*as^4*z3*beta0^2*lix*nc^2
       + 64/3*tI1*ddd*as^4*z3*beta0^2*lix^2*nc^2
       - 2*tI1*ddd*as^4*z3*beta0^3*lix*nc
       + 8/3*tI1*ddd*as^4*z3*beta0^3*lix^2*nc
       + 240*tI1*ddd*as^4*z3*z4*lix*nc^4
       - 24*tI1*ddd*as^4*z3*z4*beta0*lix*nc^3
       + 344*tI1*ddd*as^4*z3^2*lix*nc^4
       + 8*tI1*ddd*as^4*z3^2*lix^2*nc^4
       - 188*tI1*ddd*as^4*z3^2*beta0*lix*nc^3
       - 1078/3*tI1*ddd*as^4*z2*lix*nc^4
       + 7985/36*tI1*ddd*as^4*z2*lix^2*nc^4
       + 1256/3*tI1*ddd*as^4*z2*beta0*lix*nc^3
       - 14407/36*tI1*ddd*as^4*z2*beta0*lix^2*nc^3
       + 92/3*tI1*ddd*as^4*z2*beta0^2*lix*nc^2
       - 1231/9*tI1*ddd*as^4*z2*beta0^2*lix^2*nc^2
       - 10/3*tI1*ddd*as^4*z2*beta0^3*lix^2*nc
       - 80*tI1*ddd*as^4*z2*z5*lix*nc^4
       + 48*tI1*ddd*as^4*z2*z5*beta0*lix*nc^3
       - 304/3*tI1*ddd*as^4*z2*z3*lix*nc^4
       + 108*tI1*ddd*as^4*z2*z3*lix^2*nc^4
       - 200/3*tI1*ddd*as^4*z2*z3*beta0*lix*nc^3
       - 48*tI1*ddd*as^4*z2*z3*beta0*lix^2*nc^3
       + 8*tI1*ddd*as^4*z2*z3*beta0^2*lix*nc^2
       - 137131/2592*tI1*ddd*as^5*lix^3*nc^5
       + 149/24*tI1*ddd*as^5*lix^4*nc^5
       - 62297/2592*tI1*ddd*as^5*beta0*lix^3*nc^4
       - 65/8*tI1*ddd*as^5*beta0*lix^4*nc^4
       + 10547/27*tI1*ddd*as^5*beta0^2*lix^3*nc^3
       + 25/8*tI1*ddd*as^5*beta0^2*lix^4*nc^3
       - 136027/324*tI1*ddd*as^5*beta0^3*lix^3*nc^2
       - 161/24*tI1*ddd*as^5*beta0^3*lix^4*nc^2
       - 8977/648*tI1*ddd*as^5*beta0^4*lix^3*nc
       + 11/2*tI1*ddd*as^5*beta0^4*lix^4*nc
       - 1155/2*tI1*ddd*as^5*z6*lix^3*nc^5
       + 1155/2*tI1*ddd*as^5*z6*beta0*lix^3*nc^4
       + 32*tI1*ddd*as^5*z5*lix^3*nc^5
       - 64*tI1*ddd*as^5*z5*beta0*lix^3*nc^4
       + 32*tI1*ddd*as^5*z5*beta0^2*lix^3*nc^3
       + 4443/4*tI1*ddd*as^5*z4*lix^3*nc^5
       + 45/2*tI1*ddd*as^5*z4*lix^4*nc^5
       - 4369/6*tI1*ddd*as^5*z4*beta0*lix^3*nc^4
       - 45*tI1*ddd*as^5*z4*beta0*lix^4*nc^4
       - 1574/3*tI1*ddd*as^5*z4*beta0^2*lix^3*nc^3
       + 45/2*tI1*ddd*as^5*z4*beta0^2*lix^4*nc^3
       - 80/3*tI1*ddd*as^5*z4*beta0^3*lix^3*nc^2
       - 413/18*tI1*ddd*as^5*z3*lix^3*nc^5
       + 2/3*tI1*ddd*as^5*z3*lix^4*nc^5
       + 569/9*tI1*ddd*as^5*z3*beta0*lix^3*nc^4
       - 2*tI1*ddd*as^5*z3*beta0*lix^4*nc^4
       + 2023/18*tI1*ddd*as^5*z3*beta0^2*lix^3*nc^3
       + 2*tI1*ddd*as^5*z3*beta0^2*lix^4*nc^3
       - 23*tI1*ddd*as^5*z3*beta0^3*lix^3*nc^2
       - 2/3*tI1*ddd*as^5*z3*beta0^3*lix^4*nc^2
       - 5/3*tI1*ddd*as^5*z3*beta0^4*lix^3*nc
       + 180*tI1*ddd*as^5*z3*z4*lix^3*nc^5
       + 8*tI1*ddd*as^5*z3^2*lix^3*nc^5
       - 8*tI1*ddd*as^5*z3^2*beta0*lix^3*nc^4
       - 2185/36*tI1*ddd*as^5*z2*lix^3*nc^5
       - 27/2*tI1*ddd*as^5*z2*lix^4*nc^5
       - 9427/18*tI1*ddd*as^5*z2*beta0*lix^3*nc^4
       + 17*tI1*ddd*as^5*z2*beta0*lix^4*nc^4
       + 2411/4*tI1*ddd*as^5*z2*beta0^2*lix^3*nc^3
       + 6*tI1*ddd*as^5*z2*beta0^2*lix^4*nc^3
       + 1435/9*tI1*ddd*as^5*z2*beta0^3*lix^3*nc^2
       - 9*tI1*ddd*as^5*z2*beta0^3*lix^4*nc^2
       + 20/9*tI1*ddd*as^5*z2*beta0^4*lix^3*nc
       - 1/2*tI1*ddd*as^5*z2*beta0^4*lix^4*nc
       - 4*tI1*ddd*as^5*z2*z3*lix^3*nc^5
       - 228*tI1*ddd*as^5*z2*z3*beta0*lix^3*nc^4
       + 40*tI1*ddd*as^5*z2*z3*beta0^2*lix^3*nc^3
       + 8413/4320*tI1*ddd*as^6*lix^5*nc^6
       - 889/216*tI1*ddd*as^6*beta0*lix^5*nc^5
       + 1003/432*tI1*ddd*as^6*beta0^2*lix^5*nc^4
       - 377/216*tI1*ddd*as^6*beta0^3*lix^5*nc^3
       + 2809/864*tI1*ddd*as^6*beta0^4*lix^5*nc^2
       - 224/135*tI1*ddd*as^6*beta0^5*lix^5*nc
       + 15/2*tI1*ddd*as^6*z4*lix^5*nc^6
       - 45/2*tI1*ddd*as^6*z4*beta0*lix^5*nc^5
       + 45/2*tI1*ddd*as^6*z4*beta0^2*lix^5*nc^4
       - 15/2*tI1*ddd*as^6*z4*beta0^3*lix^5*nc^3
       + 1/6*tI1*ddd*as^6*z3*lix^5*nc^6
       - 2/3*tI1*ddd*as^6*z3*beta0*lix^5*nc^5
       + tI1*ddd*as^6*z3*beta0^2*lix^5*nc^4
       - 2/3*tI1*ddd*as^6*z3*beta0^3*lix^5*nc^3
       + 1/6*tI1*ddd*as^6*z3*beta0^4*lix^5*nc^2
       - 67/15*tI1*ddd*as^6*z2*lix^5*nc^6
       + 10*tI1*ddd*as^6*z2*beta0*lix^5*nc^5
       - 10/3*tI1*ddd*as^6*z2*beta0^2*lix^5*nc^4
       - 16/3*tI1*ddd*as^6*z2*beta0^3*lix^5*nc^3
       + 3*tI1*ddd*as^6*z2*beta0^4*lix^5*nc^2
       + 2/15*tI1*ddd*as^6*z2*beta0^5*lix^5*nc
       + 13/3*tI0*as*nc
       + 11/3*tI0*as*beta0
       - 6*tI0*as*z2*nc
       - 4357/72*tI0*as^2*nc^2
       + 13/3*tI0*as^2*lix*nc^2
       + 3811/72*tI0*as^2*beta0*nc
       - 2/3*tI0*as^2*beta0*lix*nc
       + 4*tI0*as^2*beta0^2
       - 11/3*tI0*as^2*beta0^2*lix
       + 14*tI0*as^2*z4*nc^2
       - 22*tI0*as^2*z3*nc^2
       + 8*tI0*as^2*z3*beta0*nc
       - 71/6*tI0*as^2*z2*nc^2
       - 6*tI0*as^2*z2*lix*nc^2
       - 38/3*tI0*as^2*z2*beta0*nc
       + 6*tI0*as^2*z2*beta0*lix*nc
       - 1175/72*tI0*as^3*lix*nc^3
       + 13/6*tI0*as^3*lix^2*nc^3
       + 3365/36*tI0*as^3*beta0*lix*nc^2
       - 5/2*tI0*as^3*beta0*lix^2*nc^2
       - 7427/72*tI0*as^3*beta0^2*lix*nc
       - 3/2*tI0*as^3*beta0^2*lix^2*nc
       - 4*tI0*as^3*beta0^3*lix
       + 11/6*tI0*as^3*beta0^3*lix^2
       + 70*tI0*as^3*z4*lix*nc^3
       - 70*tI0*as^3*z4*beta0*lix*nc^2
       - 14/3*tI0*as^3*z3*lix*nc^3
       + 146/3*tI0*as^3*z3*beta0*lix*nc^2
       - 12*tI0*as^3*z3*beta0^2*lix*nc
       - 88*tI0*as^3*z2*lix*nc^3
       - 3*tI0*as^3*z2*lix^2*nc^3
       + 153/2*tI0*as^3*z2*beta0*lix*nc^2
       + 6*tI0*as^3*z2*beta0*lix^2*nc^2
       + 34*tI0*as^3*z2*beta0^2*lix*nc
       - 3*tI0*as^3*z2*beta0^2*lix^2*nc
       - 24*tI0*as^3*z2*z3*lix*nc^3
       + 13/18*tI0*as^4*lix^3*nc^4
       - 14/9*tI0*as^4*beta0*lix^3*nc^3
       + 1/3*tI0*as^4*beta0^2*lix^3*nc^2
       + 10/9*tI0*as^4*beta0^3*lix^3*nc
       - 11/18*tI0*as^4*beta0^4*lix^3
       - tI0*as^4*z2*lix^3*nc^4
       + 3*tI0*as^4*z2*beta0*lix^3*nc^3
       - 3*tI0*as^4*z2*beta0^2*lix^3*nc^2
       + tI0*as^4*z2*beta0^3*lix^3*nc
       + 13/72*tI0*as^5*lix^4*nc^5
       - 41/72*tI0*as^5*beta0*lix^4*nc^4
       + 17/36*tI0*as^5*beta0^2*lix^4*nc^3
       + 7/36*tI0*as^5*beta0^3*lix^4*nc^2
       - 31/72*tI0*as^5*beta0^4*lix^4*nc
       + 11/72*tI0*as^5*beta0^5*lix^4
       - 1/4*tI0*as^5*z2*lix^4*nc^5
       + tI0*as^5*z2*beta0*lix^4*nc^4
       - 3/2*tI0*as^5*z2*beta0^2*lix^4*nc^3
       + tI0*as^5*z2*beta0^3*lix^4*nc^2
       - 1/4*tI0*as^5*z2*beta0^4*lix^4*nc
       + 13/360*tI0*as^6*lix^5*nc^6
       - 3/20*tI0*as^6*beta0*lix^5*nc^5
       + 5/24*tI0*as^6*beta0^2*lix^5*nc^4
       - 1/18*tI0*as^6*beta0^3*lix^5*nc^3
       - 1/8*tI0*as^6*beta0^4*lix^5*nc^2
       + 7/60*tI0*as^6*beta0^5*lix^5*nc
       - 11/360*tI0*as^6*beta0^6*lix^5
       - 1/20*tI0*as^6*z2*lix^5*nc^6
       + 1/4*tI0*as^6*z2*beta0*lix^5*nc^5
       - 1/2*tI0*as^6*z2*beta0^2*lix^5*nc^4
       + 1/2*tI0*as^6*z2*beta0^3*lix^5*nc^3
       - 1/4*tI0*as^6*z2*beta0^4*lix^5*nc^2
       + 1/20*tI0*as^6*z2*beta0^5*lix^5*nc
       - 314227/432*tI0*ddd*as^3*nc^3
       + 6094/27*tI0*ddd*as^3*beta0*nc^2
       + 2547/16*tI0*ddd*as^3*beta0^2*nc
       + 4/3*tI0*ddd*as^3*beta0^3
       - 241*tI0*ddd*as^3*z6*nc^3
       + 93*tI0*ddd*as^3*z6*beta0*nc^2
       + 260*tI0*ddd*as^3*z5*nc^3
       - 118*tI0*ddd*as^3*z5*beta0*nc^2
       + 146*tI0*ddd*as^3*z4*nc^3
       + 164*tI0*ddd*as^3*z4*beta0*nc^2
       - 7*tI0*ddd*as^3*z4*beta0^2*nc
       - 589/6*tI0*ddd*as^3*z3*nc^3
       - 154*tI0*ddd*as^3*z3*beta0*nc^2
       + 200/3*tI0*ddd*as^3*z3*beta0^2*nc
       - 2*tI0*ddd*as^3*z3*beta0^3
       - 48*tI0*ddd*as^3*z3^2*nc^3
       + 24*tI0*ddd*as^3*z3^2*beta0*nc^2
       + 3245/12*tI0*ddd*as^3*z2*nc^3
       - 5375/18*tI0*ddd*as^3*z2*beta0*nc^2
       - 281/9*tI0*ddd*as^3*z2*beta0^2*nc
       + 72*tI0*ddd*as^3*z2*z3*nc^3
       - 20*tI0*ddd*as^3*z2*z3*beta0*nc^2
       - 668677/864*tI0*ddd*as^4*lix*nc^4
       + 33479/1296*tI0*ddd*as^4*lix^2*nc^4
       + 2162975/864*tI0*ddd*as^4*beta0*lix*nc^3
       + 25159/432*tI0*ddd*as^4*beta0*lix^2*nc^3
       - 591635/432*tI0*ddd*as^4*beta0^2*lix*nc^2
       - 35989/432*tI0*ddd*as^4*beta0^2*lix^2*nc^2
       - 39727/144*tI0*ddd*as^4*beta0^3*lix*nc
       + 107011/1296*tI0*ddd*as^4*beta0^3*lix^2*nc
       - 4/3*tI0*ddd*as^4*beta0^4*lix
       + 2*tI0*ddd*as^4*beta0^4*lix^2
       - 880*tI0*ddd*as^4*z6*lix*nc^4
       - 315/2*tI0*ddd*as^4*z6*lix^2*nc^4
       + 1036*tI0*ddd*as^4*z6*beta0*lix*nc^3
       - 93*tI0*ddd*as^4*z6*beta0^2*lix*nc^2
       + 1660/3*tI0*ddd*as^4*z5*lix*nc^4
       - 502/3*tI0*ddd*as^4*z5*beta0*lix*nc^3
       + 126*tI0*ddd*as^4*z5*beta0^2*lix*nc^2
       + 7499/6*tI0*ddd*as^4*z4*lix*nc^4
       + 260*tI0*ddd*as^4*z4*lix^2*nc^4
       - 3223/3*tI0*ddd*as^4*z4*beta0*lix*nc^3
       + 35*tI0*ddd*as^4*z4*beta0*lix^2*nc^3
       - 677*tI0*ddd*as^4*z4*beta0^2*lix*nc^2
       + 65*tI0*ddd*as^4*z4*beta0^2*lix^2*nc^2
       + 11*tI0*ddd*as^4*z4*beta0^3*lix*nc
       - 12497/18*tI0*ddd*as^4*z3*lix*nc^4
       + 7*tI0*ddd*as^4*z3*lix^2*nc^4
       + 735*tI0*ddd*as^4*z3*beta0*lix*nc^3
       + 70/3*tI0*ddd*as^4*z3*beta0*lix^2*nc^3
       + 994/9*tI0*ddd*as^4*z3*beta0^2*lix*nc^2
       - 107/3*tI0*ddd*as^4*z3*beta0^2*lix^2*nc^2
       - 90*tI0*ddd*as^4*z3*beta0^3*lix*nc
       + 16/3*tI0*ddd*as^4*z3*beta0^3*lix^2*nc
       + 2*tI0*ddd*as^4*z3*beta0^4*lix
       + 400*tI0*ddd*as^4*z3*z4*lix*nc^4
       - 144*tI0*ddd*as^4*z3^2*lix*nc^4
       + 128*tI0*ddd*as^4*z3^2*beta0*lix*nc^3
       - 24*tI0*ddd*as^4*z3^2*beta0^2*lix*nc^2
       - 4227/8*tI0*ddd*as^4*z2*lix*nc^4
       - 551/4*tI0*ddd*as^4*z2*lix^2*nc^4
       - 5683/12*tI0*ddd*as^4*z2*beta0*lix*nc^3
       + 197/3*tI0*ddd*as^4*z2*beta0*lix^2*nc^3
       + 1195*tI0*ddd*as^4*z2*beta0^2*lix*nc^2
       - 1147/12*tI0*ddd*as^4*z2*beta0^2*lix^2*nc^2
       + 211/3*tI0*ddd*as^4*z2*beta0^3*lix*nc
       - 73/3*tI0*ddd*as^4*z2*beta0^3*lix^2*nc
       - 384*tI0*ddd*as^4*z2*z5*lix*nc^4
       + 508/3*tI0*ddd*as^4*z2*z3*lix*nc^4
       - 24*tI0*ddd*as^4*z2*z3*lix^2*nc^4
       - 1792/3*tI0*ddd*as^4*z2*z3*beta0*lix*nc^3
       + 24*tI0*ddd*as^4*z2*z3*beta0*lix^2*nc^3
       + 100*tI0*ddd*as^4*z2*z3*beta0^2*lix*nc^2
       + 31775/1296*tI0*ddd*as^5*lix^3*nc^5
       + 2371/324*tI0*ddd*as^5*beta0*lix^3*nc^4
       - 9661/216*tI0*ddd*as^5*beta0^2*lix^3*nc^3
       + 17527/324*tI0*ddd*as^5*beta0^3*lix^3*nc^2
       - 52537/1296*tI0*ddd*as^5*beta0^4*lix^3*nc
       - 2/3*tI0*ddd*as^5*beta0^5*lix^3
       - 315/2*tI0*ddd*as^5*z6*lix^3*nc^5
       + 315/2*tI0*ddd*as^5*z6*beta0*lix^3*nc^4
       + 680/3*tI0*ddd*as^5*z4*lix^3*nc^5
       - 125*tI0*ddd*as^5*z4*beta0*lix^3*nc^4
       - 70*tI0*ddd*as^5*z4*beta0^2*lix^3*nc^3
       - 95/3*tI0*ddd*as^5*z4*beta0^3*lix^3*nc^2
       + 16/3*tI0*ddd*as^5*z3*lix^3*nc^5
       + 5/3*tI0*ddd*as^5*z3*beta0*lix^3*nc^4
       - 21*tI0*ddd*as^5*z3*beta0^2*lix^3*nc^3
       + 47/3*tI0*ddd*as^5*z3*beta0^3*lix^3*nc^2
       - 5/3*tI0*ddd*as^5*z3*beta0^4*lix^3*nc
       - 1727/18*tI0*ddd*as^5*z2*lix^3*nc^5
       + 2899/36*tI0*ddd*as^5*z2*beta0*lix^3*nc^4
       - 164/3*tI0*ddd*as^5*z2*beta0^2*lix^3*nc^3
       + 2143/36*tI0*ddd*as^5*z2*beta0^3*lix^3*nc^2
       + 95/9*tI0*ddd*as^5*z2*beta0^4*lix^3*nc
       - 12*tI0*ddd*as^5*z2*z3*lix^3*nc^5
       + 24*tI0*ddd*as^5*z2*z3*beta0*lix^3*nc^4
       - 12*tI0*ddd*as^5*z2*z3*beta0^2*lix^3*nc^3
      ;

* eq.(3.18)
L   P3nsLnc =
       + 16/81*L0^3*[1-x]^(-1)*cf*nf^3
       - 212/27*L0^3*[1-x]^(-1)*cf*nc*nf^2
       + 725/9*L0^3*[1-x]^(-1)*cf*nc^2*nf
       - 55291/324*L0^3*[1-x]^(-1)*cf*nc^3
       - 8/81*L0^3*cf*nf^3
       + 92/27*L0^3*cf*nc*nf^2
       - 851/27*L0^3*cf*nc^2*nf
       + 2987/81*L0^3*cf*nc^3
       - 8/81*L0^3*x*cf*nf^3
       + 20/3*L0^3*x*cf*nc*nf^2
       - 2257/27*L0^3*x*cf*nc^2*nf
       + 38641/162*L0^3*x*cf*nc^3
       + 32/3*L0^3*z3*[1-x]^(-1)*cf*nc^3
       - 28/3*L0^3*z3*cf*nc^3
       - 28/3*L0^3*z3*x*cf*nc^3
       - 32*L0^3*z2*[1-x]^(-1)*cf*nc^2*nf
       + 84*L0^3*z2*[1-x]^(-1)*cf*nc^3
       + 24*L0^3*z2*cf*nc^2*nf
       - 52*L0^3*z2*cf*nc^3
       + 24*L0^3*z2*x*cf*nc^2*nf
       - 340/3*L0^3*z2*x*cf*nc^3
       - 8/9*L0^4*[1-x]^(-1)*cf*nc*nf^2
       + 92/9*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 227/6*L0^4*[1-x]^(-1)*cf*nc^3
       + 2/3*L0^4*cf*nc*nf^2
       - 65/9*L0^4*cf*nc^2*nf
       + 463/18*L0^4*cf*nc^3
       + 2/3*L0^4*x*cf*nc*nf^2
       - 31/3*L0^4*x*cf*nc^2*nf
       + 251/6*L0^4*x*cf*nc^3
       + 56/3*L0^4*z2*[1-x]^(-1)*cf*nc^3
       - 49/3*L0^4*z2*cf*nc^3
       - 49/3*L0^4*z2*x*cf*nc^3
       + 8/9*L0^5*[1-x]^(-1)*cf*nc^2*nf
       - 26/9*L0^5*[1-x]^(-1)*cf*nc^3
       - 7/9*L0^5*cf*nc^2*nf
       + 22/9*L0^5*cf*nc^3
       - 7/9*L0^5*x*cf*nc^2*nf
       + 34/9*L0^5*x*cf*nc^3
       - 2/9*L0^6*[1-x]^(-1)*cf*nc^3
       + 5/24*L0^6*cf*nc^3
       + 5/24*L0^6*x*cf*nc^3
       - 16/3*H(R(0,0,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       + 4/3*H(R(0,0,1),x)*L0^3*cf*nc^3
       + 4/3*H(R(0,0,1),x)*L0^3*x*cf*nc^3
       - 80/9*H(R(0,1),x)*L0^3*[1-x]^(-1)*cf*nc^2*nf
       + 188/9*H(R(0,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       + 56/9*H(R(0,1),x)*L0^3*cf*nc^2*nf
       - 50/9*H(R(0,1),x)*L0^3*cf*nc^3
       + 56/9*H(R(0,1),x)*L0^3*x*cf*nc^2*nf
       - 218/9*H(R(0,1),x)*L0^3*x*cf*nc^3
       + 8/3*H(R(0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 2/3*H(R(0,1),x)*L0^4*cf*nc^3
       - 2/3*H(R(0,1),x)*L0^4*x*cf*nc^3
       + 112/3*H(R(0,1,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       - 88/3*H(R(0,1,1),x)*L0^3*cf*nc^3
       - 88/3*H(R(0,1,1),x)*L0^3*x*cf*nc^3
       + 16/27*H(R(1),x)*L0^3*[1-x]^(-1)*cf*nc*nf^2
       + 724/27*H(R(1),x)*L0^3*[1-x]^(-1)*cf*nc^2*nf
       - 4745/27*H(R(1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       - 8/27*H(R(1),x)*L0^3*cf*nc*nf^2
       - 338/27*H(R(1),x)*L0^3*cf*nc^2*nf
       + 3977/54*H(R(1),x)*L0^3*cf*nc^3
       - 8/27*H(R(1),x)*L0^3*x*cf*nc*nf^2
       - 386/27*H(R(1),x)*L0^3*x*cf*nc^2*nf
       + 5513/54*H(R(1),x)*L0^3*x*cf*nc^3
       + 368/3*H(R(1),x)*L0^3*z2*[1-x]^(-1)*cf*nc^3
       - 184/3*H(R(1),x)*L0^3*z2*cf*nc^3
       - 184/3*H(R(1),x)*L0^3*z2*x*cf*nc^3
       + 32/9*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 140/9*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 16/9*H(R(1),x)*L0^4*cf*nc^2*nf
       + 58/9*H(R(1),x)*L0^4*cf*nc^3
       - 16/9*H(R(1),x)*L0^4*x*cf*nc^2*nf
       + 82/9*H(R(1),x)*L0^4*x*cf*nc^3
       - 8/3*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       + 4/3*H(R(1),x)*L0^5*cf*nc^3
       + 4/3*H(R(1),x)*L0^5*x*cf*nc^3
       + 112/3*H(R(1,0,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       - 56/3*H(R(1,0,1),x)*L0^3*cf*nc^3
       - 56/3*H(R(1,0,1),x)*L0^3*x*cf*nc^3
       - 160/9*H(R(1,1),x)*L0^3*[1-x]^(-1)*cf*nc^2*nf
       + 160/9*H(R(1,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       + 80/9*H(R(1,1),x)*L0^3*cf*nc^2*nf
       + 160/9*H(R(1,1),x)*L0^3*cf*nc^3
       + 80/9*H(R(1,1),x)*L0^3*x*cf*nc^2*nf
       - 320/9*H(R(1,1),x)*L0^3*x*cf*nc^3
       - 16/3*H(R(1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 8/3*H(R(1,1),x)*L0^4*cf*nc^3
       + 8/3*H(R(1,1),x)*L0^4*x*cf*nc^3
       + 320/3*H(R(1,1,1),x)*L0^3*[1-x]^(-1)*cf*nc^3
       - 160/3*H(R(1,1,1),x)*L0^3*cf*nc^3
       - 160/3*H(R(1,1,1),x)*L0^3*x*cf*nc^3
      ;

* eq.(3.19)
L   P4nsLnc =
       + 16/81*L0^5*[1-x]^(-1)*cf*nc*nf^3
       - 44/9*L0^5*[1-x]^(-1)*cf*nc^2*nf^2
       + 1091/27*L0^5*[1-x]^(-1)*cf*nc^3*nf
       - 26449/324*L0^5*[1-x]^(-1)*cf*nc^4
       - 4/27*L0^5*cf*nc*nf^3
       + 94/27*L0^5*cf*nc^2*nf^2
       - 745/27*L0^5*cf*nc^3*nf
       + 10643/216*L0^5*cf*nc^4
       - 4/27*L0^5*x*cf*nc*nf^3
       + 122/27*L0^5*x*cf*nc^2*nf^2
       - 1183/27*L0^5*x*cf*nc^3*nf
       + 8129/72*L0^5*x*cf*nc^4
       + 16/3*L0^5*z3*[1-x]^(-1)*cf*nc^4
       - 5*L0^5*z3*cf*nc^4
       - 5*L0^5*z3*x*cf*nc^4
       - 16*L0^5*z2*[1-x]^(-1)*cf*nc^3*nf
       + 44*L0^5*z2*[1-x]^(-1)*cf*nc^4
       + 14*L0^5*z2*cf*nc^3*nf
       - 36*L0^5*z2*cf*nc^4
       + 14*L0^5*z2*x*cf*nc^3*nf
       - 196/3*L0^5*z2*x*cf*nc^4
       - 8/27*L0^6*[1-x]^(-1)*cf*nc^2*nf^2
       + 236/81*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 1525/162*L0^6*[1-x]^(-1)*cf*nc^4
       + 7/27*L0^6*cf*nc^2*nf^2
       - 67/27*L0^6*cf*nc^3*nf
       + 283/36*L0^6*cf*nc^4
       + 7/27*L0^6*x*cf*nc^2*nf^2
       - 89/27*L0^6*x*cf*nc^3*nf
       + 641/54*L0^6*x*cf*nc^4
       + 40/9*L0^6*z2*[1-x]^(-1)*cf*nc^4
       - 25/6*L0^6*z2*cf*nc^4
       - 25/6*L0^6*z2*x*cf*nc^4
       + 4/27*L0^7*[1-x]^(-1)*cf*nc^3*nf
       - 13/27*L0^7*[1-x]^(-1)*cf*nc^4
       - 5/36*L0^7*cf*nc^3*nf
       + 4/9*L0^7*cf*nc^4
       - 5/36*L0^7*x*cf*nc^3*nf
       + 2/3*L0^7*x*cf*nc^4
       - 1/45*L0^8*[1-x]^(-1)*cf*nc^4
       + 31/1440*L0^8*cf*nc^4
       + 31/1440*L0^8*x*cf*nc^4
       + 16/3*H(R(0,0,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       - 16/3*H(R(0,0,1),x)*L0^5*cf*nc^4
       - 16/3*H(R(0,0,1),x)*L0^5*x*cf*nc^4
       - 16/3*H(R(0,1),x)*L0^5*[1-x]^(-1)*cf*nc^3*nf
       + 28/3*H(R(0,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       + 10/3*H(R(0,1),x)*L0^5*cf*nc^3*nf
       + 2/3*H(R(0,1),x)*L0^5*cf*nc^4
       + 10/3*H(R(0,1),x)*L0^5*x*cf*nc^3*nf
       - 38/3*H(R(0,1),x)*L0^5*x*cf*nc^4
       + 1/3*H(R(0,1),x)*L0^6*cf*nc^4
       + 1/3*H(R(0,1),x)*L0^6*x*cf*nc^4
       + 80/3*H(R(0,1,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       - 56/3*H(R(0,1,1),x)*L0^5*cf*nc^4
       - 56/3*H(R(0,1,1),x)*L0^5*x*cf*nc^4
       - 8/27*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^2*nf^2
       + 148/9*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^3*nf
       - 2314/27*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       + 4/27*H(R(1),x)*L0^5*cf*nc^2*nf^2
       - 74/9*H(R(1),x)*L0^5*cf*nc^3*nf
       + 1049/27*H(R(1),x)*L0^5*cf*nc^4
       + 4/27*H(R(1),x)*L0^5*x*cf*nc^2*nf^2
       - 74/9*H(R(1),x)*L0^5*x*cf*nc^3*nf
       + 1265/27*H(R(1),x)*L0^5*x*cf*nc^4
       + 176/3*H(R(1),x)*L0^5*z2*[1-x]^(-1)*cf*nc^4
       - 88/3*H(R(1),x)*L0^5*z2*cf*nc^4
       - 88/3*H(R(1),x)*L0^5*z2*x*cf*nc^4
       + 4/3*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 16/3*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 2/3*H(R(1),x)*L0^6*cf*nc^3*nf
       + 2*H(R(1),x)*L0^6*cf*nc^4
       - 2/3*H(R(1),x)*L0^6*x*cf*nc^3*nf
       + 10/3*H(R(1),x)*L0^6*x*cf*nc^4
       - 4/9*H(R(1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       + 2/9*H(R(1),x)*L0^7*cf*nc^4
       + 2/9*H(R(1),x)*L0^7*x*cf*nc^4
       + 80/3*H(R(1,0,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       - 40/3*H(R(1,0,1),x)*L0^5*cf*nc^4
       - 40/3*H(R(1,0,1),x)*L0^5*x*cf*nc^4
       - 64/9*H(R(1,1),x)*L0^5*[1-x]^(-1)*cf*nc^3*nf
       - 8/9*H(R(1,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       + 32/9*H(R(1,1),x)*L0^5*cf*nc^3*nf
       + 124/9*H(R(1,1),x)*L0^5*cf*nc^4
       + 32/9*H(R(1,1),x)*L0^5*x*cf*nc^3*nf
       - 116/9*H(R(1,1),x)*L0^5*x*cf*nc^4
       - 8/3*H(R(1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       + 4/3*H(R(1,1),x)*L0^6*cf*nc^4
       + 4/3*H(R(1,1),x)*L0^6*x*cf*nc^4
       + 160/3*H(R(1,1,1),x)*L0^5*[1-x]^(-1)*cf*nc^4
       - 80/3*H(R(1,1,1),x)*L0^5*cf*nc^4
       - 80/3*H(R(1,1,1),x)*L0^5*x*cf*nc^4
      ;

* -----------------------------------------------------------------------------
*
* Section 4
*
* -----------------------------------------------------------------------------

* eq.(4.1)
L   C2nsR =
       + 5*as*cf
       + 2*as*z2*cf
       + F
       + 5111/9216*F*as*cf^(-1)*beta0^2
       + 5/6*F*as*ca
       + 85/48*F*as*beta0
       - 125/8*F*as*cf
       + 75/4*F*as*z2*cf^(-1)*ca^2
       - 60*F*as*z2*ca
       + 48*F*as*z2*cf
       - 79/1152*F^3*as*cf^(-1)*beta0^2
       - 5/64*F^3*as*beta0
       + 1/2*F^3*as*cf
       - 2093/9216*F^5*as*cf^(-1)*beta0^2
       + 5/6*F^5*as*ca
       + 9/8*F^5*as*beta0
       + 9/16*F^5*as*cf
       - 15/4*F^5*as*z2*cf^(-1)*ca^2
       + 12*F^5*as*z2*ca
       - 12*F^5*as*z2*cf
       - 77/576*F^7*as*cf^(-1)*beta0^2
       + 3/32*F^7*as*beta0
       + 181/9216*F^9*as*cf^(-1)*beta0^2
       - 5/16*F^9*as*beta0
       + 5/16*F^9*as*cf
       + 35/384*F^11*as*cf^(-1)*beta0^2
       - 35/192*F^11*as*beta0
       + 385/9216*F^13*as*cf^(-1)*beta0^2
       - 11/64*N*F^(-1)*cf^(-1)*beta0
       - 1/2*N*F^(-1)
       + 11/48*N*F*cf^(-1)*beta0
       + 3/8*N*F
       + 1/32*N*F^3*cf^(-1)*beta0
       - 1/16*N*F^5*cf^(-1)*beta0
       + 1/8*N*F^5
       - 5/192*N*F^7*cf^(-1)*beta0
       + 5/72*N^2*F^(-3)*cf^(-2)*beta0^2
       + 5/12*N^2*F^(-3)*cf^(-1)*ca
       + 29/48*N^2*F^(-3)*cf^(-1)*beta0
       - 37/16*N^2*F^(-3)
       + 15/4*N^2*F^(-3)*z2*cf^(-2)*ca^2
       - 12*N^2*F^(-3)*z2*cf^(-1)*ca
       + 19/2*N^2*F^(-3)*z2
       - 5/72*N^2*F^(-1)*cf^(-2)*beta0^2
       - 5/12*N^2*F^(-1)*cf^(-1)*ca
       - 29/48*N^2*F^(-1)*cf^(-1)*beta0
       + 37/16*N^2*F^(-1)
       - 15/4*N^2*F^(-1)*z2*cf^(-2)*ca^2
       + 12*N^2*F^(-1)*z2*cf^(-1)*ca
       - 19/2*N^2*F^(-1)*z2
      ;

* eq.(4.2)
L   CLnsR =
       + 76/9*as^2*cf*nf
       - 538/9*as^2*cf*ca
       + 74*as^2*cf^2
       + 8*as^2*z2*cf^2
       + 4*F*as*cf
       - 1321/2304*F*as^2*beta0^2
       + 50/3*F*as^2*cf*ca
       + 20*F*as^2*cf*beta0
       - 193/2*F*as^2*cf^2
       + 15*F*as^2*z2*ca^2
       - 48*F*as^2*z2*cf*ca
       + 32*F*as^2*z2*cf^2
       + 53/288*F^3*as^2*beta0^2
       - 115/48*F^3*as^2*cf*beta0
       + 4*F^3*as^2*cf^2
       - 269/2304*F^5*as^2*beta0^2
       + 10/3*F^5*as^2*cf*ca
       + 47/12*F^5*as^2*cf*beta0
       + 1/4*F^5*as^2*cf^2
       - 15*F^5*as^2*z2*ca^2
       + 48*F^5*as^2*z2*cf*ca
       - 48*F^5*as^2*z2*cf^2
       - 47/144*F^7*as^2*beta0^2
       + 19/24*F^7*as^2*cf*beta0
       + 181/2304*F^9*as^2*beta0^2
       - 5/4*F^9*as^2*cf*beta0
       + 5/4*F^9*as^2*cf^2
       + 35/96*F^11*as^2*beta0^2
       - 35/48*F^11*as^2*cf*beta0
       + 385/2304*F^13*as^2*beta0^2
       + 5/16*N*F^(-1)*as*beta0
       - 4*N*F^(-1)*as*cf
       + 4*N*as*cf
       - 1/12*N*F*as*beta0
       - 1/2*N*F*as*cf
       + 1/8*N*F^3*as*beta0
       - 1/4*N*F^5*as*beta0
       + 1/2*N*F^5*as*cf
       - 5/48*N*F^7*as*beta0
       - 1/18*N^2*F^(-3)*as*cf^(-1)*beta0^2
       + 5/3*N^2*F^(-3)*as*ca
       + 23/12*N^2*F^(-3)*as*beta0
       - 25/4*N^2*F^(-3)*as*cf
       - 2*N^2*F^(-3)*as*z2*cf
       + 1/18*N^2*F^(-1)*as*cf^(-1)*beta0^2
       - 5/3*N^2*F^(-1)*as*ca
       - 23/12*N^2*F^(-1)*as*beta0
       + 41/4*N^2*F^(-1)*as*cf
       + 2*N^2*F^(-1)*as*z2*cf
       - 4*N^2*as*cf
      ;

* eq.(4.3)
L   C3nsR =
       + 7*as*cf
       + 2*as*z2*cf
       + F
       + 5111/9216*F*as*cf^(-1)*beta0^2
       + 5/6*F*as*ca
       + 3/8*F*as*beta0
       - 157/8*F*as*cf
       + 75/4*F*as*z2*cf^(-1)*ca^2
       - 60*F*as*z2*ca
       + 48*F*as*z2*cf
       - 79/1152*F^3*as*cf^(-1)*beta0^2
       + 29/192*F^3*as*beta0
       - 2093/9216*F^5*as*cf^(-1)*beta0^2
       + 5/6*F^5*as*ca
       + 73/48*F^5*as*beta0
       + 1/16*F^5*as*cf
       - 15/4*F^5*as*z2*cf^(-1)*ca^2
       + 12*F^5*as*z2*ca
       - 12*F^5*as*z2*cf
       - 77/576*F^7*as*cf^(-1)*beta0^2
       + 19/96*F^7*as*beta0
       + 181/9216*F^9*as*cf^(-1)*beta0^2
       - 5/16*F^9*as*beta0
       + 5/16*F^9*as*cf
       + 35/384*F^11*as*cf^(-1)*beta0^2
       - 35/192*F^11*as*beta0
       + 385/9216*F^13*as*cf^(-1)*beta0^2
       - 11/64*N*F^(-1)*cf^(-1)*beta0
       + 11/48*N*F*cf^(-1)*beta0
       - 1/8*N*F
       + 1/32*N*F^3*cf^(-1)*beta0
       - 1/16*N*F^5*cf^(-1)*beta0
       + 1/8*N*F^5
       - 5/192*N*F^7*cf^(-1)*beta0
       + 5/72*N^2*F^(-3)*cf^(-2)*beta0^2
       + 5/12*N^2*F^(-3)*cf^(-1)*ca
       + 7/16*N^2*F^(-3)*cf^(-1)*beta0
       - 49/16*N^2*F^(-3)
       + 15/4*N^2*F^(-3)*z2*cf^(-2)*ca^2
       - 12*N^2*F^(-3)*z2*cf^(-1)*ca
       + 19/2*N^2*F^(-3)*z2
       - 5/72*N^2*F^(-1)*cf^(-2)*beta0^2
       - 5/12*N^2*F^(-1)*cf^(-1)*ca
       - 7/16*N^2*F^(-1)*cf^(-1)*beta0
       + 49/16*N^2*F^(-1)
       - 15/4*N^2*F^(-1)*z2*cf^(-2)*ca^2
       + 12*N^2*F^(-1)*z2*cf^(-1)*ca
       - 19/2*N^2*F^(-1)*z2
      ;

* eq.(4.5)
L   C2ns4R =
       + 390*N^(-8)*cf^4
       - 1822/3*N^(-7)*cf^3*beta0
       + 1052*N^(-7)*cf^4
       + 1951/6*N^(-6)*cf^2*beta0^2
       + 2180/3*N^(-6)*cf^3*ca
       - 448*N^(-6)*cf^3*beta0
       + 336*N^(-6)*cf^4
       - 1560*N^(-6)*z2*cf^2*ca^2
       + 4992*N^(-6)*z2*cf^3*ca
       - 5872*N^(-6)*z2*cf^4
      ;

* eq.(4.6)
L   CLns4R =
       + 240*N^(-6)*cf^4
       - 992/3*N^(-5)*cf^3*beta0
       + 472*N^(-5)*cf^4
       + 460/3*N^(-4)*cf^2*beta0^2
       + 480*N^(-4)*cf^3*ca
       + 56*N^(-4)*cf^3*beta0
       - 644*N^(-4)*cf^4
       - 1200*N^(-4)*z2*cf^2*ca^2
       + 3840*N^(-4)*z2*cf^3*ca
       - 4016*N^(-4)*z2*cf^4
      ;

* eq.(4.7)
L   C3ns4R =
       + 390*N^(-8)*cf^4
       - 1822/3*N^(-7)*cf^3*beta0
       + 780*N^(-7)*cf^4
       + 1951/6*N^(-6)*cf^2*beta0^2
       + 2180/3*N^(-6)*cf^3*ca
       - 8/3*N^(-6)*cf^3*beta0
       - 496*N^(-6)*cf^4
       - 1560*N^(-6)*z2*cf^2*ca^2
       + 4992*N^(-6)*z2*cf^3*ca
       - 5872*N^(-6)*z2*cf^4
      ;

* eq.(4.8)
L   C2ns5R =
       + 2652*N^(-10)*cf^5
       - 17012/3*N^(-9)*cf^4*beta0
       + 8418*N^(-9)*cf^5
       + 14363/3*N^(-8)*cf^3*beta0^2
       + 6040*N^(-8)*cf^4*ca
       - 23546/3*N^(-8)*cf^4*beta0
       + 6438*N^(-8)*cf^5
       - 15840*N^(-8)*z2*cf^3*ca^2
       + 50688*N^(-8)*z2*cf^4*ca
       - 56508*N^(-8)*z2*cf^5
      ;

* eq.(4.9)
L   CLns5R =
       + 1560*N^(-8)*cf^5
       - 8920/3*N^(-7)*cf^4*beta0
       + 3736*N^(-7)*cf^5
       + 6574/3*N^(-6)*cf^3*beta0^2
       + 11120/3*N^(-6)*cf^4*ca
       - 1504*N^(-6)*cf^4*beta0
       - 2064*N^(-6)*cf^5
       - 10800*N^(-6)*z2*cf^3*ca^2
       + 34560*N^(-6)*z2*cf^4*ca
       - 35648*N^(-6)*z2*cf^5
      ;

* eq.(4.10)
L   C3ns5R =
       + 2652*N^(-10)*cf^5
       - 17012/3*N^(-9)*cf^4*beta0
       + 6630*N^(-9)*cf^5
       + 14363/3*N^(-8)*cf^3*beta0^2
       + 6040*N^(-8)*cf^4*ca
       - 11374/3*N^(-8)*cf^4*beta0
       + 66*N^(-8)*cf^5
       - 15840*N^(-8)*z2*cf^3*ca^2
       + 50688*N^(-8)*z2*cf^4*ca
       - 56508*N^(-8)*z2*cf^5
      ;

* eq.(4.11)
L   C24nsLnc =
       + 119/81*L0^4*[1-x]^(-1)*cf*nf^3
       - 20567/486*L0^4*[1-x]^(-1)*cf*nc*nf^2
       + 1323835/3888*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 2033255/2592*L0^4*[1-x]^(-1)*cf*nc^3
       - 119/162*L0^4*cf*nf^3
       + 111/4*L0^4*cf*nc*nf^2
       - 461075/1944*L0^4*cf*nc^2*nf
       + 42910871/77760*L0^4*cf*nc^3
       - 119/162*L0^4*x*cf*nf^3
       + 11249/324*L0^4*x*cf*nc*nf^2
       - 338959/972*L0^4*x*cf*nc^2*nf
       + 74167091/77760*L0^4*x*cf*nc^3
       - 16*L0^4*x^2*cf*nc^2*nf
       + 264/5*L0^4*x^2*cf*nc^3
       + 175/3*L0^4*z3*[1-x]^(-1)*cf*nc^3
       - 467/9*L0^4*z3*cf*nc^3
       - 467/9*L0^4*z3*x*cf*nc^3
       - 76*L0^4*z2*[1-x]^(-1)*cf*nc^2*nf
       + 1100/3*L0^4*z2*[1-x]^(-1)*cf*nc^3
       + 3245/54*L0^4*z2*cf*nc^2*nf
       - 32143/108*L0^4*z2*cf*nc^3
       + 3125/54*L0^4*z2*x*cf*nc^2*nf
       - 41005/108*L0^4*z2*x*cf*nc^3
       + 32/3*L0^4*z2*x^2*cf*nc^2*nf
       - 32*L0^4*z2*x^2*cf*nc^3
       - 16*L0^4*z2*x^3*cf*nc^2*nf
       + 264/5*L0^4*z2*x^3*cf*nc^3
       - 1951/810*L0^5*[1-x]^(-1)*cf*nc*nf^2
       + 24083/810*L0^5*[1-x]^(-1)*cf*nc^2*nf
       - 1131721/12960*L0^5*[1-x]^(-1)*cf*nc^3
       + 1951/1080*L0^5*cf*nc*nf^2
       - 8543/360*L0^5*cf*nc^2*nf
       + 152641/2160*L0^5*cf*nc^3
       + 1951/1080*L0^5*x*cf*nc*nf^2
       - 3433/120*L0^5*x*cf*nc^2*nf
       + 54733/540*L0^5*x*cf*nc^3
       + 20*L0^5*z2*[1-x]^(-1)*cf*nc^3
       - 1067/60*L0^5*z2*cf*nc^3
       - 1067/60*L0^5*z2*x*cf*nc^3
       + 911/810*L0^6*[1-x]^(-1)*cf*nc^2*nf
       - 17531/3240*L0^6*[1-x]^(-1)*cf*nc^3
       - 6377/6480*L0^6*cf*nc^2*nf
       + 6247/1296*L0^6*cf*nc^3
       - 6377/6480*L0^6*x*cf*nc^2*nf
       + 38867/6480*L0^6*x*cf*nc^3
       - 13/84*L0^7*[1-x]^(-1)*cf*nc^3
       + 65/448*L0^7*cf*nc^3
       + 65/448*L0^7*x*cf*nc^3
       - 8/3*H(R(0,0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 53/18*H(R(0,0,1),x)*L0^4*cf*nc^3
       - 53/18*H(R(0,0,1),x)*L0^4*x*cf*nc^3
       - 220/27*H(R(0,1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       + 1573/108*H(R(0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 10/9*H(R(0,1),x)*L0^4*cf*nc^2*nf
       + 69/2*H(R(0,1),x)*L0^4*cf*nc^3
       - 22/9*H(R(0,1),x)*L0^4*x*cf*nc^2*nf
       + 85/2*H(R(0,1),x)*L0^4*x*cf*nc^3
       + 32/3*H(R(0,1),x)*L0^4*x^2*cf*nc^2*nf
       - 32*H(R(0,1),x)*L0^4*x^2*cf*nc^3
       - 16*H(R(0,1),x)*L0^4*x^3*cf*nc^2*nf
       + 264/5*H(R(0,1),x)*L0^4*x^3*cf*nc^3
       - 37/30*H(R(0,1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       + 329/120*H(R(0,1),x)*L0^5*cf*nc^3
       + 329/120*H(R(0,1),x)*L0^5*x*cf*nc^3
       + 208/9*H(R(0,1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 85/18*H(R(0,1,1),x)*L0^4*cf*nc^3
       - 85/18*H(R(0,1,1),x)*L0^4*x*cf*nc^3
       + 16/9*H(R(1),x)*L0^4*x^(-1)*cf*nc^2*nf
       - 88/15*H(R(1),x)*L0^4*x^(-1)*cf*nc^3
       - 671/81*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc*nf^2
       + 2797/18*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 186223/324*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 671/162*H(R(1),x)*L0^4*cf*nc*nf^2
       - 8177/108*H(R(1),x)*L0^4*cf*nc^2*nf
       + 808271/3240*H(R(1),x)*L0^4*cf*nc^3
       + 671/162*H(R(1),x)*L0^4*x*cf*nc*nf^2
       - 383/4*H(R(1),x)*L0^4*x*cf*nc^2*nf
       + 1282259/3240*H(R(1),x)*L0^4*x*cf*nc^3
       + 16*H(R(1),x)*L0^4*x^2*cf*nc^2*nf
       - 264/5*H(R(1),x)*L0^4*x^2*cf*nc^3
       + 1852/9*H(R(1),x)*L0^4*z2*[1-x]^(-1)*cf*nc^3
       - 926/9*H(R(1),x)*L0^4*z2*cf*nc^3
       - 926/9*H(R(1),x)*L0^4*z2*x*cf*nc^3
       + 1324/135*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^2*nf
       - 30613/540*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       - 662/135*H(R(1),x)*L0^5*cf*nc^2*nf
       + 29137/1080*H(R(1),x)*L0^5*cf*nc^3
       - 662/135*H(R(1),x)*L0^5*x*cf*nc^2*nf
       + 35869/1080*H(R(1),x)*L0^5*x*cf*nc^3
       - 229/90*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^3
       + 229/180*H(R(1),x)*L0^6*cf*nc^3
       + 229/180*H(R(1),x)*L0^6*x*cf*nc^3
       + 53/3*H(R(1,0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 53/6*H(R(1,0,1),x)*L0^4*cf*nc^3
       - 53/6*H(R(1,0,1),x)*L0^4*x*cf*nc^3
       - 16/9*H(R(1,1),x)*L0^4*x^(-2)*cf*nc^2*nf
       + 88/15*H(R(1,1),x)*L0^4*x^(-2)*cf*nc^3
       + 844/27*H(R(1,1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 7432/27*H(R(1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 422/27*H(R(1,1),x)*L0^4*cf*nc^2*nf
       + 8323/54*H(R(1,1),x)*L0^4*cf*nc^3
       - 518/27*H(R(1,1),x)*L0^4*x*cf*nc^2*nf
       + 8833/54*H(R(1,1),x)*L0^4*x*cf*nc^3
       + 32/3*H(R(1,1),x)*L0^4*x^2*cf*nc^2*nf
       - 32*H(R(1,1),x)*L0^4*x^2*cf*nc^3
       - 16*H(R(1,1),x)*L0^4*x^3*cf*nc^2*nf
       + 264/5*H(R(1,1),x)*L0^4*x^3*cf*nc^3
       - 19*H(R(1,1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       + 19/2*H(R(1,1),x)*L0^5*cf*nc^3
       + 19/2*H(R(1,1),x)*L0^5*x*cf*nc^3
       - 284/9*H(R(1,1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 142/9*H(R(1,1,1),x)*L0^4*cf*nc^3
       + 142/9*H(R(1,1,1),x)*L0^4*x*cf*nc^3
      ;

* five-loop result corresponding to eq.(4.11)
L   C25nsLnc =
       + 28243/18225*L0^6*[1-x]^(-1)*cf*nc*nf^3
       - 4534891/145800*L0^6*[1-x]^(-1)*cf*nc^2*nf^2
       + 59215931/291600*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 970160333/2332800*L0^6*[1-x]^(-1)*cf*nc^4
       - 28243/24300*L0^6*cf*nc*nf^3
       + 14393303/583200*L0^6*cf*nc^2*nf^2
       - 24069691/145800*L0^6*cf*nc^3*nf
       + 396968701/1166400*L0^6*cf*nc^4
       - 28243/24300*L0^6*x*cf*nc*nf^3
       + 16745003/583200*L0^6*x*cf*nc^2*nf^2
       - 128758339/583200*L0^6*x*cf*nc^3*nf
       + 1228142507/2332800*L0^6*x*cf*nc^4
       - 58/15*L0^6*x^2*cf*nc^3*nf
       + 319/25*L0^6*x^2*cf*nc^4
       + 73/5*L0^6*z3*[1-x]^(-1)*cf*nc^4
       - 119077/8640*L0^6*z3*cf*nc^4
       - 119077/8640*L0^6*z3*x*cf*nc^4
       - 4261/135*L0^6*z2*[1-x]^(-1)*cf*nc^3*nf
       + 159227/1080*L0^6*z2*[1-x]^(-1)*cf*nc^4
       + 36401/1296*L0^6*z2*cf*nc^3*nf
       - 6836981/51840*L0^6*z2*cf*nc^4
       + 36053/1296*L0^6*z2*x*cf*nc^3*nf
       - 8567189/51840*L0^6*z2*x*cf*nc^4
       + 116/45*L0^6*z2*x^2*cf*nc^3*nf
       - 116/15*L0^6*z2*x^2*cf*nc^4
       - 58/15*L0^6*z2*x^3*cf*nc^3*nf
       + 319/25*L0^6*z2*x^3*cf*nc^4
       - 14363/17010*L0^7*[1-x]^(-1)*cf*nc^2*nf^2
       + 315509/34020*L0^7*[1-x]^(-1)*cf*nc^3*nf
       - 6900599/272160*L0^7*[1-x]^(-1)*cf*nc^4
       + 14363/19440*L0^7*cf*nc^2*nf^2
       - 448681/54432*L0^7*cf*nc^3*nf
       + 4930715/217728*L0^7*cf*nc^4
       + 14363/19440*L0^7*x*cf*nc^2*nf^2
       - 377303/38880*L0^7*x*cf*nc^3*nf
       + 17016227/544320*L0^7*x*cf*nc^4
       + 169/42*L0^7*z2*[1-x]^(-1)*cf*nc^4
       - 25499/6720*L0^7*z2*cf*nc^4
       - 25499/6720*L0^7*z2*x*cf*nc^4
       + 4253/22680*L0^8*[1-x]^(-1)*cf*nc^3*nf
       - 160483/181440*L0^8*[1-x]^(-1)*cf*nc^4
       - 4253/24192*L0^8*cf*nc^3*nf
       + 806159/967680*L0^8*cf*nc^4
       - 4253/24192*L0^8*x*cf*nc^3*nf
       + 981371/967680*L0^8*x*cf*nc^4
       - 221/15120*L0^9*[1-x]^(-1)*cf*nc^4
       + 6851/483840*L0^9*cf*nc^4
       + 6851/483840*L0^9*x*cf*nc^4
       + 67/15*H(R(0,0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 9937/2160*H(R(0,0,1),x)*L0^6*cf*nc^4
       - 9937/2160*H(R(0,0,1),x)*L0^6*x*cf*nc^4
       - 232/135*H(R(0,1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 4031/540*H(R(0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 101/90*H(R(0,1),x)*L0^6*cf*nc^3*nf
       + 106721/4320*H(R(0,1),x)*L0^6*cf*nc^4
       - 107/54*H(R(0,1),x)*L0^6*x*cf*nc^3*nf
       + 97169/4320*H(R(0,1),x)*L0^6*x*cf*nc^4
       + 116/45*H(R(0,1),x)*L0^6*x^2*cf*nc^3*nf
       - 116/15*H(R(0,1),x)*L0^6*x^2*cf*nc^4
       - 58/15*H(R(0,1),x)*L0^6*x^3*cf*nc^3*nf
       + 319/25*H(R(0,1),x)*L0^6*x^3*cf*nc^4
       - 169/210*H(R(0,1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       + 809/840*H(R(0,1),x)*L0^7*cf*nc^4
       + 809/840*H(R(0,1),x)*L0^7*x*cf*nc^4
       + 1633/135*H(R(0,1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 4699/1080*H(R(0,1,1),x)*L0^6*cf*nc^4
       - 4699/1080*H(R(0,1,1),x)*L0^6*x*cf*nc^4
       + 58/135*H(R(1),x)*L0^6*x^(-1)*cf*nc^3*nf
       - 319/225*H(R(1),x)*L0^6*x^(-1)*cf*nc^4
       - 17113/2430*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^2*nf^2
       + 157889/1620*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 2434069/7776*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       + 17113/4860*H(R(1),x)*L0^6*cf*nc^2*nf^2
       - 6283/135*H(R(1),x)*L0^6*cf*nc^3*nf
       + 52228459/388800*H(R(1),x)*L0^6*cf*nc^4
       + 17113/4860*H(R(1),x)*L0^6*x*cf*nc^2*nf^2
       - 45149/810*H(R(1),x)*L0^6*x*cf*nc^3*nf
       + 77644411/388800*H(R(1),x)*L0^6*x*cf*nc^4
       + 58/15*H(R(1),x)*L0^6*x^2*cf*nc^3*nf
       - 319/25*H(R(1),x)*L0^6*x^2*cf*nc^4
       + 3913/54*H(R(1),x)*L0^6*z2*[1-x]^(-1)*cf*nc^4
       - 3913/108*H(R(1),x)*L0^6*z2*cf*nc^4
       - 3913/108*H(R(1),x)*L0^6*z2*x*cf*nc^4
       + 2789/945*H(R(1),x)*L0^7*[1-x]^(-1)*cf*nc^3*nf
       - 3419/216*H(R(1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       - 2789/1890*H(R(1),x)*L0^7*cf*nc^3*nf
       + 54743/7560*H(R(1),x)*L0^7*cf*nc^4
       - 2789/1890*H(R(1),x)*L0^7*x*cf*nc^3*nf
       + 69017/7560*H(R(1),x)*L0^7*x*cf*nc^4
       - 209/560*H(R(1),x)*L0^8*[1-x]^(-1)*cf*nc^4
       + 209/1120*H(R(1),x)*L0^8*cf*nc^4
       + 209/1120*H(R(1),x)*L0^8*x*cf*nc^4
       + 491/45*H(R(1,0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 491/90*H(R(1,0,1),x)*L0^6*cf*nc^4
       - 491/90*H(R(1,0,1),x)*L0^6*x*cf*nc^4
       - 58/135*H(R(1,1),x)*L0^6*x^(-2)*cf*nc^3*nf
       + 319/225*H(R(1,1),x)*L0^6*x^(-2)*cf*nc^4
       + 2383/135*H(R(1,1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 18217/135*H(R(1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 2383/270*H(R(1,1),x)*L0^6*cf*nc^3*nf
       + 78991/1080*H(R(1,1),x)*L0^6*cf*nc^4
       - 523/54*H(R(1,1),x)*L0^6*x*cf*nc^3*nf
       + 5261/72*H(R(1,1),x)*L0^6*x*cf*nc^4
       + 116/45*H(R(1,1),x)*L0^6*x^2*cf*nc^3*nf
       - 116/15*H(R(1,1),x)*L0^6*x^2*cf*nc^4
       - 58/15*H(R(1,1),x)*L0^6*x^3*cf*nc^3*nf
       + 319/25*H(R(1,1),x)*L0^6*x^3*cf*nc^4
       - 1007/210*H(R(1,1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       + 1007/420*H(R(1,1),x)*L0^7*cf*nc^4
       + 1007/420*H(R(1,1),x)*L0^7*x*cf*nc^4
       - 704/135*H(R(1,1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       + 352/135*H(R(1,1,1),x)*L0^6*cf*nc^4
       + 352/135*H(R(1,1,1),x)*L0^6*x*cf*nc^4
      ;

* eq.(4.12)
L   C34nsLnc =
       + 119/81*L0^4*[1-x]^(-1)*cf*nf^3
       - 20567/486*L0^4*[1-x]^(-1)*cf*nc*nf^2
       + 1323835/3888*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 2033255/2592*L0^4*[1-x]^(-1)*cf*nc^3
       - 119/162*L0^4*cf*nf^3
       + 8239/324*L0^4*cf*nc*nf^2
       - 208597/972*L0^4*cf*nc^2*nf
       + 7672909/15552*L0^4*cf*nc^3
       - 119/162*L0^4*x*cf*nf^3
       + 3499/108*L0^4*x*cf*nc*nf^2
       - 633641/1944*L0^4*x*cf*nc^2*nf
       + 14040109/15552*L0^4*x*cf*nc^3
       + 175/3*L0^4*z3*[1-x]^(-1)*cf*nc^3
       - 467/9*L0^4*z3*cf*nc^3
       - 467/9*L0^4*z3*x*cf*nc^3
       - 76*L0^4*z2*[1-x]^(-1)*cf*nc^2*nf
       + 1100/3*L0^4*z2*[1-x]^(-1)*cf*nc^3
       + 3245/54*L0^4*z2*cf*nc^2*nf
       - 62873/216*L0^4*z2*cf*nc^3
       + 3221/54*L0^4*z2*x*cf*nc^2*nf
       - 82325/216*L0^4*z2*x*cf*nc^3
       - 16/9*L0^4*z2*x^2*cf*nc^2*nf
       + 88/9*L0^4*z2*x^2*cf*nc^3
       - 1951/810*L0^5*[1-x]^(-1)*cf*nc*nf^2
       + 24083/810*L0^5*[1-x]^(-1)*cf*nc^2*nf
       - 1131721/12960*L0^5*[1-x]^(-1)*cf*nc^3
       + 1951/1080*L0^5*cf*nc*nf^2
       - 24961/1080*L0^5*cf*nc^2*nf
       + 9811/144*L0^5*cf*nc^3
       + 1951/1080*L0^5*x*cf*nc*nf^2
       - 30229/1080*L0^5*x*cf*nc^2*nf
       + 7135/72*L0^5*x*cf*nc^3
       + 20*L0^5*z2*[1-x]^(-1)*cf*nc^3
       - 1067/60*L0^5*z2*cf*nc^3
       - 1067/60*L0^5*z2*x*cf*nc^3
       + 911/810*L0^6*[1-x]^(-1)*cf*nc^2*nf
       - 17531/3240*L0^6*[1-x]^(-1)*cf*nc^3
       - 6377/6480*L0^6*cf*nc^2*nf
       + 30929/6480*L0^6*cf*nc^3
       - 6377/6480*L0^6*x*cf*nc^2*nf
       + 38561/6480*L0^6*x*cf*nc^3
       - 13/84*L0^7*[1-x]^(-1)*cf*nc^3
       + 65/448*L0^7*cf*nc^3
       + 65/448*L0^7*x*cf*nc^3
       - 8/3*H(R(0,0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 53/18*H(R(0,0,1),x)*L0^4*cf*nc^3
       - 53/18*H(R(0,0,1),x)*L0^4*x*cf*nc^3
       - 220/27*H(R(0,1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       + 1573/108*H(R(0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 10/9*H(R(0,1),x)*L0^4*cf*nc^2*nf
       + 65/2*H(R(0,1),x)*L0^4*cf*nc^3
       + 10/9*H(R(0,1),x)*L0^4*x*cf*nc^2*nf
       + 49/2*H(R(0,1),x)*L0^4*x*cf*nc^3
       - 16/9*H(R(0,1),x)*L0^4*x^2*cf*nc^2*nf
       + 88/9*H(R(0,1),x)*L0^4*x^2*cf*nc^3
       - 37/30*H(R(0,1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       + 329/120*H(R(0,1),x)*L0^5*cf*nc^3
       + 329/120*H(R(0,1),x)*L0^5*x*cf*nc^3
       + 208/9*H(R(0,1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 85/18*H(R(0,1,1),x)*L0^4*cf*nc^3
       - 85/18*H(R(0,1,1),x)*L0^4*x*cf*nc^3
       - 671/81*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc*nf^2
       + 2797/18*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 186223/324*H(R(1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 671/162*H(R(1),x)*L0^4*cf*nc*nf^2
       - 8065/108*H(R(1),x)*L0^4*cf*nc^2*nf
       + 159931/648*H(R(1),x)*L0^4*cf*nc^3
       + 671/162*H(R(1),x)*L0^4*x*cf*nc*nf^2
       - 9461/108*H(R(1),x)*L0^4*x*cf*nc^2*nf
       + 235015/648*H(R(1),x)*L0^4*x*cf*nc^3
       + 1852/9*H(R(1),x)*L0^4*z2*[1-x]^(-1)*cf*nc^3
       - 926/9*H(R(1),x)*L0^4*z2*cf*nc^3
       - 926/9*H(R(1),x)*L0^4*z2*x*cf*nc^3
       + 1324/135*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^2*nf
       - 30613/540*H(R(1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       - 662/135*H(R(1),x)*L0^5*cf*nc^2*nf
       + 28057/1080*H(R(1),x)*L0^5*cf*nc^3
       - 662/135*H(R(1),x)*L0^5*x*cf*nc^2*nf
       + 34789/1080*H(R(1),x)*L0^5*x*cf*nc^3
       - 229/90*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^3
       + 229/180*H(R(1),x)*L0^6*cf*nc^3
       + 229/180*H(R(1),x)*L0^6*x*cf*nc^3
       + 53/3*H(R(1,0,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 53/6*H(R(1,0,1),x)*L0^4*cf*nc^3
       - 53/6*H(R(1,0,1),x)*L0^4*x*cf*nc^3
       + 16/9*H(R(1,1),x)*L0^4*x^(-1)*cf*nc^2*nf
       - 88/9*H(R(1,1),x)*L0^4*x^(-1)*cf*nc^3
       + 844/27*H(R(1,1),x)*L0^4*[1-x]^(-1)*cf*nc^2*nf
       - 7432/27*H(R(1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       - 422/27*H(R(1,1),x)*L0^4*cf*nc^2*nf
       + 7963/54*H(R(1,1),x)*L0^4*cf*nc^3
       - 422/27*H(R(1,1),x)*L0^4*x*cf*nc^2*nf
       + 7609/54*H(R(1,1),x)*L0^4*x*cf*nc^3
       - 16/9*H(R(1,1),x)*L0^4*x^2*cf*nc^2*nf
       + 88/9*H(R(1,1),x)*L0^4*x^2*cf*nc^3
       - 19*H(R(1,1),x)*L0^5*[1-x]^(-1)*cf*nc^3
       + 19/2*H(R(1,1),x)*L0^5*cf*nc^3
       + 19/2*H(R(1,1),x)*L0^5*x*cf*nc^3
       - 284/9*H(R(1,1,1),x)*L0^4*[1-x]^(-1)*cf*nc^3
       + 142/9*H(R(1,1,1),x)*L0^4*cf*nc^3
       + 142/9*H(R(1,1,1),x)*L0^4*x*cf*nc^3
      ;

* five-loop result corresponding to eq.(4.12)
L   C35nsLnc =
       + 28243/18225*L0^6*[1-x]^(-1)*cf*nc*nf^3
       - 4534891/145800*L0^6*[1-x]^(-1)*cf*nc^2*nf^2
       + 59215931/291600*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 970160333/2332800*L0^6*[1-x]^(-1)*cf*nc^4
       - 28243/24300*L0^6*cf*nc*nf^3
       + 14066573/583200*L0^6*cf*nc^2*nf^2
       - 5817439/36450*L0^6*cf*nc^3*nf
       + 380794903/1166400*L0^6*cf*nc^4
       - 28243/24300*L0^6*x*cf*nc*nf^3
       + 16418273/583200*L0^6*x*cf*nc^2*nf^2
       - 125797819/583200*L0^6*x*cf*nc^3*nf
       + 1202946131/2332800*L0^6*x*cf*nc^4
       + 73/5*L0^6*z3*[1-x]^(-1)*cf*nc^4
       - 119077/8640*L0^6*z3*cf*nc^4
       - 119077/8640*L0^6*z3*x*cf*nc^4
       - 4261/135*L0^6*z2*[1-x]^(-1)*cf*nc^3*nf
       + 159227/1080*L0^6*z2*[1-x]^(-1)*cf*nc^4
       + 36401/1296*L0^6*z2*cf*nc^3*nf
       - 6787661/51840*L0^6*z2*cf*nc^4
       + 181657/6480*L0^6*z2*x*cf*nc^3*nf
       - 8567981/51840*L0^6*z2*x*cf*nc^4
       - 58/135*L0^6*z2*x^2*cf*nc^3*nf
       + 319/135*L0^6*z2*x^2*cf*nc^4
       - 14363/17010*L0^7*[1-x]^(-1)*cf*nc^2*nf^2
       + 315509/34020*L0^7*[1-x]^(-1)*cf*nc^3*nf
       - 6900599/272160*L0^7*[1-x]^(-1)*cf*nc^4
       + 14363/19440*L0^7*cf*nc^2*nf^2
       - 2225147/272160*L0^7*cf*nc^3*nf
       + 24337921/1088640*L0^7*cf*nc^4
       + 14363/19440*L0^7*x*cf*nc^2*nf^2
       - 2622863/272160*L0^7*x*cf*nc^3*nf
       + 8455849/272160*L0^7*x*cf*nc^4
       + 169/42*L0^7*z2*[1-x]^(-1)*cf*nc^4
       - 25499/6720*L0^7*z2*cf*nc^4
       - 25499/6720*L0^7*z2*x*cf*nc^4
       + 4253/22680*L0^8*[1-x]^(-1)*cf*nc^3*nf
       - 160483/181440*L0^8*[1-x]^(-1)*cf*nc^4
       - 4253/24192*L0^8*cf*nc^3*nf
       + 803477/967680*L0^8*cf*nc^4
       - 4253/24192*L0^8*x*cf*nc^3*nf
       + 978689/967680*L0^8*x*cf*nc^4
       - 221/15120*L0^9*[1-x]^(-1)*cf*nc^4
       + 6851/483840*L0^9*cf*nc^4
       + 6851/483840*L0^9*x*cf*nc^4
       + 67/15*H(R(0,0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 9937/2160*H(R(0,0,1),x)*L0^6*cf*nc^4
       - 9937/2160*H(R(0,0,1),x)*L0^6*x*cf*nc^4
       - 232/135*H(R(0,1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 4031/540*H(R(0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 101/90*H(R(0,1),x)*L0^6*cf*nc^3*nf
       + 104033/4320*H(R(0,1),x)*L0^6*cf*nc^4
       - 101/90*H(R(0,1),x)*L0^6*x*cf*nc^3*nf
       + 77777/4320*H(R(0,1),x)*L0^6*x*cf*nc^4
       - 58/135*H(R(0,1),x)*L0^6*x^2*cf*nc^3*nf
       + 319/135*H(R(0,1),x)*L0^6*x^2*cf*nc^4
       - 169/210*H(R(0,1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       + 809/840*H(R(0,1),x)*L0^7*cf*nc^4
       + 809/840*H(R(0,1),x)*L0^7*x*cf*nc^4
       + 1633/135*H(R(0,1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 4699/1080*H(R(0,1,1),x)*L0^6*cf*nc^4
       - 4699/1080*H(R(0,1,1),x)*L0^6*x*cf*nc^4
       - 17113/2430*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^2*nf^2
       + 157889/1620*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 2434069/7776*H(R(1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       + 17113/4860*H(R(1),x)*L0^6*cf*nc^2*nf^2
       - 37279/810*H(R(1),x)*L0^6*cf*nc^3*nf
       + 10298519/77760*H(R(1),x)*L0^6*cf*nc^4
       + 17113/4860*H(R(1),x)*L0^6*x*cf*nc^2*nf^2
       - 7223/135*H(R(1),x)*L0^6*x*cf*nc^3*nf
       + 14820167/77760*H(R(1),x)*L0^6*x*cf*nc^4
       + 3913/54*H(R(1),x)*L0^6*z2*[1-x]^(-1)*cf*nc^4
       - 3913/108*H(R(1),x)*L0^6*z2*cf*nc^4
       - 3913/108*H(R(1),x)*L0^6*z2*x*cf*nc^4
       + 2789/945*H(R(1),x)*L0^7*[1-x]^(-1)*cf*nc^3*nf
       - 3419/216*H(R(1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       - 2789/1890*H(R(1),x)*L0^7*cf*nc^3*nf
       + 53573/7560*H(R(1),x)*L0^7*cf*nc^4
       - 2789/1890*H(R(1),x)*L0^7*x*cf*nc^3*nf
       + 67847/7560*H(R(1),x)*L0^7*x*cf*nc^4
       - 209/560*H(R(1),x)*L0^8*[1-x]^(-1)*cf*nc^4
       + 209/1120*H(R(1),x)*L0^8*cf*nc^4
       + 209/1120*H(R(1),x)*L0^8*x*cf*nc^4
       + 491/45*H(R(1,0,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 491/90*H(R(1,0,1),x)*L0^6*cf*nc^4
       - 491/90*H(R(1,0,1),x)*L0^6*x*cf*nc^4
       + 58/135*H(R(1,1),x)*L0^6*x^(-1)*cf*nc^3*nf
       - 319/135*H(R(1,1),x)*L0^6*x^(-1)*cf*nc^4
       + 2383/135*H(R(1,1),x)*L0^6*[1-x]^(-1)*cf*nc^3*nf
       - 18217/135*H(R(1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       - 2383/270*H(R(1,1),x)*L0^6*cf*nc^3*nf
       + 76939/1080*H(R(1,1),x)*L0^6*cf*nc^4
       - 2383/270*H(R(1,1),x)*L0^6*x*cf*nc^3*nf
       + 24229/360*H(R(1,1),x)*L0^6*x*cf*nc^4
       - 58/135*H(R(1,1),x)*L0^6*x^2*cf*nc^3*nf
       + 319/135*H(R(1,1),x)*L0^6*x^2*cf*nc^4
       - 1007/210*H(R(1,1),x)*L0^7*[1-x]^(-1)*cf*nc^4
       + 1007/420*H(R(1,1),x)*L0^7*cf*nc^4
       + 1007/420*H(R(1,1),x)*L0^7*x*cf*nc^4
       - 704/135*H(R(1,1,1),x)*L0^6*[1-x]^(-1)*cf*nc^4
       + 352/135*H(R(1,1,1),x)*L0^6*cf*nc^4
       + 352/135*H(R(1,1,1),x)*L0^6*x*cf*nc^4
      ;

* eq.(4.13)
L   CL4nsLnc =
       - 34/27*L0^4*cf*nc^2*nf
       + 1687/1080*L0^4*cf*nc^3
       + 376/81*L0^4*x*cf*nc*nf^2
       - 16361/324*L0^4*x*cf*nc^2*nf
       + 867587/6480*L0^4*x*cf*nc^3
       - 32/3*L0^4*x^2*cf*nc^2*nf
       + 176/5*L0^4*x^2*cf*nc^3
       - 16/9*L0^4*z2*x*cf*nc^2*nf
       - 61/12*L0^4*z2*x*cf*nc^3
       + 64/9*L0^4*z2*x^2*cf*nc^2*nf
       - 64/3*L0^4*z2*x^2*cf*nc^3
       - 32/3*L0^4*z2*x^3*cf*nc^2*nf
       + 176/5*L0^4*z2*x^3*cf*nc^3
       - 1/4*L0^5*cf*nc^3
       - 167/135*L0^5*x*cf*nc^2*nf
       + 1693/270*L0^5*x*cf*nc^3
       + 17/180*L0^6*x*cf*nc^3
       - 32/9*H(R(0,1),x)*L0^4*x*cf*nc^2*nf
       + 20*H(R(0,1),x)*L0^4*x*cf*nc^3
       + 64/9*H(R(0,1),x)*L0^4*x^2*cf*nc^2*nf
       - 64/3*H(R(0,1),x)*L0^4*x^2*cf*nc^3
       - 32/3*H(R(0,1),x)*L0^4*x^3*cf*nc^2*nf
       + 176/5*H(R(0,1),x)*L0^4*x^3*cf*nc^3
       + 64/9*H(R(1),x)*L0^4*x^(-1)*cf*nc^2*nf
       - 352/15*H(R(1),x)*L0^4*x^(-1)*cf*nc^3
       - 116/15*H(R(1),x)*L0^4*cf*nc^3
       - 248/27*H(R(1),x)*L0^4*x*cf*nc^2*nf
       + 5869/135*H(R(1),x)*L0^4*x*cf*nc^3
       + 32/3*H(R(1),x)*L0^4*x^2*cf*nc^2*nf
       - 176/5*H(R(1),x)*L0^4*x^2*cf*nc^3
       + 2*H(R(1),x)*L0^5*x*cf*nc^3
       - 64/9*H(R(1,1),x)*L0^4*x^(-2)*cf*nc^2*nf
       + 352/15*H(R(1,1),x)*L0^4*x^(-2)*cf*nc^3
       + 32/9*H(R(1,1),x)*L0^4*x^(-1)*cf*nc^2*nf
       - 32/3*H(R(1,1),x)*L0^4*x^(-1)*cf*nc^3
       - 32/9*H(R(1,1),x)*L0^4*x*cf*nc^2*nf
       + 88/3*H(R(1,1),x)*L0^4*x*cf*nc^3
       + 64/9*H(R(1,1),x)*L0^4*x^2*cf*nc^2*nf
       - 64/3*H(R(1,1),x)*L0^4*x^2*cf*nc^3
       - 32/3*H(R(1,1),x)*L0^4*x^3*cf*nc^2*nf
       + 176/5*H(R(1,1),x)*L0^4*x^3*cf*nc^3
      ;

* eq.(4.14)
L   CL5nsLnc =
       - 1669/3240*L0^6*cf*nc^3*nf
       + 82109/64800*L0^6*cf*nc^4
       + 10891/9720*L0^6*x*cf*nc^2*nf^2
       - 23149/1944*L0^6*x*cf*nc^3*nf
       + 6243947/194400*L0^6*x*cf*nc^4
       - 116/45*L0^6*x^2*cf*nc^3*nf
       + 638/75*L0^6*x^2*cf*nc^4
       - 29/135*L0^6*z2*x*cf*nc^3*nf
       - 337/360*L0^6*z2*x*cf*nc^4
       + 232/135*L0^6*z2*x^2*cf*nc^3*nf
       - 232/45*L0^6*z2*x^2*cf*nc^4
       - 116/45*L0^6*z2*x^3*cf*nc^3*nf
       + 638/75*L0^6*z2*x^3*cf*nc^4
       - 13/672*L0^7*cf*nc^4
       - 3043/22680*L0^7*x*cf*nc^3*nf
       + 3823/5670*L0^7*x*cf*nc^4
       + 149/26880*L0^8*x*cf*nc^4
       - 116/135*H(R(0,1),x)*L0^6*x*cf*nc^3*nf
       + 46/9*H(R(0,1),x)*L0^6*x*cf*nc^4
       + 232/135*H(R(0,1),x)*L0^6*x^2*cf*nc^3*nf
       - 232/45*H(R(0,1),x)*L0^6*x^2*cf*nc^4
       - 116/45*H(R(0,1),x)*L0^6*x^3*cf*nc^3*nf
       + 638/75*H(R(0,1),x)*L0^6*x^3*cf*nc^4
       + 232/135*H(R(1),x)*L0^6*x^(-1)*cf*nc^3*nf
       - 1276/225*H(R(1),x)*L0^6*x^(-1)*cf*nc^4
       - 971/450*H(R(1),x)*L0^6*cf*nc^4
       - 223/81*H(R(1),x)*L0^6*x*cf*nc^3*nf
       + 106633/8100*H(R(1),x)*L0^6*x*cf*nc^4
       + 116/45*H(R(1),x)*L0^6*x^2*cf*nc^3*nf
       - 638/75*H(R(1),x)*L0^6*x^2*cf*nc^4
       + 13/42*H(R(1),x)*L0^7*x*cf*nc^4
       - 232/135*H(R(1,1),x)*L0^6*x^(-2)*cf*nc^3*nf
       + 1276/225*H(R(1,1),x)*L0^6*x^(-2)*cf*nc^4
       + 116/135*H(R(1,1),x)*L0^6*x^(-1)*cf*nc^3*nf
       - 116/45*H(R(1,1),x)*L0^6*x^(-1)*cf*nc^4
       - 116/135*H(R(1,1),x)*L0^6*x*cf*nc^3*nf
       + 23/3*H(R(1,1),x)*L0^6*x*cf*nc^4
       + 232/135*H(R(1,1),x)*L0^6*x^2*cf*nc^3*nf
       - 232/45*H(R(1,1),x)*L0^6*x^2*cf*nc^4
       - 116/45*H(R(1,1),x)*L0^6*x^3*cf*nc^3*nf
       + 638/75*H(R(1,1),x)*L0^6*x^3*cf*nc^4
      ;

* -----------------------------------------------------------------------------
*
* Section 5
*
* -----------------------------------------------------------------------------

* eq.(5.9)
L   Pqq3R =
       - 640*N^(-7)*cf*ca^2*nf
       + 320*N^(-7)*cf^2*nf^2
       + 640*N^(-7)*cf^2*ca*nf
       - 480*N^(-7)*cf^3*nf
       + 80*N^(-7)*cf^4
       - 2176/3*N^(-6)*cf*ca^2*nf
       - 256*N^(-6)*cf^2*nf^2
       + 3424/3*N^(-6)*cf^2*ca*nf
       - 1024/3*N^(-6)*cf^3*nf
       - 80*N^(-6)*cf^3*beta0
       + 160*N^(-6)*cf^4
       - 32/9*N^(-5)*cf*nf^3
       - 288*N^(-5)*cf*ca*nf^2
       - 13672/9*N^(-5)*cf*ca^2*nf
       + 15232/9*N^(-5)*cf^2*nf^2
       + 4328/9*N^(-5)*cf^2*ca*nf
       + 24*N^(-5)*cf^2*beta0^2
       - 1384*N^(-5)*cf^3*nf
       + 160*N^(-5)*cf^3*ca
       + 80*N^(-5)*cf^3*beta0
       + 128*N^(-5)*cf^4
       + 512/3*N^(-5)*z2*cf*ca^2*nf
       - 1184*N^(-5)*z2*cf^2*ca*nf
       - 480*N^(-5)*z2*cf^2*ca^2
       + 4192/3*N^(-5)*z2*cf^3*nf
       + 1536*N^(-5)*z2*cf^3*ca
       - 1600*N^(-5)*z2*cf^4
      ;

* eq.(5.10)
L   Pqg3R =
       - 640*N^(-7)*ca^3*nf
       + 640*N^(-7)*cf*ca*nf^2
       + 320*N^(-7)*cf*ca^2*nf
       - 320*N^(-7)*cf^2*nf^2
       - 160*N^(-7)*cf^2*ca*nf
       + 80*N^(-7)*cf^3*nf
       - 320/3*N^(-6)*ca^2*nf^2
       - 416/3*N^(-6)*ca^3*nf
       - 1408/3*N^(-6)*cf*ca*nf^2
       + 192*N^(-6)*cf*ca^2*nf
       + 432*N^(-6)*cf^2*nf^2
       - 632/3*N^(-6)*cf^2*ca*nf
       - 32/3*N^(-6)*cf^3*nf
       - 32/9*N^(-5)*ca*nf^3
       - 4736/27*N^(-5)*ca^2*nf^2
       - 68440/27*N^(-5)*ca^3*nf
       + 2224/27*N^(-5)*cf*nf^3
       + 23608/9*N^(-5)*cf*ca*nf^2
       + 51416/27*N^(-5)*cf*ca^2*nf
       - 54332/27*N^(-5)*cf^2*nf^2
       - 13414/27*N^(-5)*cf^2*ca*nf
       + 1114/3*N^(-5)*cf^3*nf
       - 96*N^(-5)*z2*ca^2*nf^2
       - 160*N^(-5)*z2*ca^3*nf
       + 800/3*N^(-5)*z2*cf*ca*nf^2
       - 3520/3*N^(-5)*z2*cf*ca^2*nf
       + 1600/3*N^(-5)*z2*cf^2*nf^2
       + 4304/3*N^(-5)*z2*cf^2*ca*nf
       - 2896/3*N^(-5)*z2*cf^3*nf
      ;

* eq.(5.11)
L   Pgq3R =
       + 1280*N^(-7)*cf*ca^3
       - 1280*N^(-7)*cf^2*ca*nf
       - 640*N^(-7)*cf^2*ca^2
       + 640*N^(-7)*cf^3*nf
       + 320*N^(-7)*cf^3*ca
       - 160*N^(-7)*cf^4
       + 640/3*N^(-6)*cf*ca^2*nf
       + 4160/3*N^(-6)*cf*ca^3
       - 640*N^(-6)*cf^2*ca*nf
       - 1280*N^(-6)*cf^2*ca^2
       + 800/3*N^(-6)*cf^3*nf
       + 2800/3*N^(-6)*cf^3*ca
       - 320*N^(-6)*cf^4
       + 64/9*N^(-5)*cf*ca*nf^2
       + 34688/27*N^(-5)*cf*ca^2*nf
       + 61936/27*N^(-5)*cf*ca^3
       - 12256/27*N^(-5)*cf^2*nf^2
       - 13072/3*N^(-5)*cf^2*ca*nf
       + 31504/27*N^(-5)*cf^2*ca^2
       + 61976/27*N^(-5)*cf^3*nf
       - 34532/27*N^(-5)*cf^3*ca
       - 100/3*N^(-5)*cf^4
       + 192*N^(-5)*z2*cf*ca^2*nf
       + 4160/3*N^(-5)*z2*cf*ca^3
       - 2624/3*N^(-5)*z2*cf^2*ca*nf
       + 1664/3*N^(-5)*z2*cf^2*ca^2
       - 2176/3*N^(-5)*z2*cf^3*nf
       - 5632/3*N^(-5)*z2*cf^3*ca
       + 1664*N^(-5)*z2*cf^4
      ;

* eq.(5.12)
L   Pgg3R =
       + 1280*N^(-7)*ca^4
       - 1920*N^(-7)*cf*ca^2*nf
       + 320*N^(-7)*cf^2*nf^2
       + 640*N^(-7)*cf^2*ca*nf
       - 160*N^(-7)*cf^3*nf
       + 1280/3*N^(-6)*ca^3*nf
       + 640/3*N^(-6)*ca^4
       - 640/3*N^(-6)*cf*ca*nf^2
       + 1856/3*N^(-6)*cf*ca^2*nf
       - 1472/3*N^(-6)*cf^2*nf^2
       + 256/3*N^(-6)*cf^2*ca*nf
       + 64/3*N^(-6)*cf^3*nf
       + 128/3*N^(-5)*ca^2*nf^2
       + 2560/3*N^(-5)*ca^3*nf
       + 4384*N^(-5)*ca^4
       - 32/9*N^(-5)*cf*nf^3
       - 4768/9*N^(-5)*cf*ca*nf^2
       - 7336*N^(-5)*cf*ca^2*nf
       + 19904/9*N^(-5)*cf^2*nf^2
       + 15976/9*N^(-5)*cf^2*ca*nf
       - 520*N^(-5)*cf^3*nf
       + 384*N^(-5)*z2*ca^3*nf
       + 2048*N^(-5)*z2*ca^4
       - 5504/3*N^(-5)*z2*cf*ca^2*nf
       - 672*N^(-5)*z2*cf^2*ca*nf
       + 1184/3*N^(-5)*z2*cf^3*nf
      ;

* eq.(5.13)
L   Pqq4R =
       + 7168*N^(-9)*cf*ca^3*nf
       - 7168*N^(-9)*cf^2*ca*nf^2
       - 7168*N^(-9)*cf^2*ca^2*nf
       + 5376*N^(-9)*cf^3*nf^2
       + 5376*N^(-9)*cf^3*ca*nf
       - 3584*N^(-9)*cf^4*nf
       + 448*N^(-9)*cf^5
       + 1792/3*N^(-8)*cf*ca^2*nf^2
       + 7936*N^(-8)*cf*ca^3*nf
       + 896/3*N^(-8)*cf^2*nf^3
       - 1088*N^(-8)*cf^2*ca*nf^2
       - 38720/3*N^(-8)*cf^2*ca^2*nf
       - 6656/3*N^(-8)*cf^3*nf^2
       + 41984/3*N^(-8)*cf^3*ca*nf
       - 12272/3*N^(-8)*cf^4*nf
       - 560*N^(-8)*cf^4*beta0
       + 1120*N^(-8)*cf^5
       + 256/9*N^(-7)*cf*ca*nf^3
       + 14144/3*N^(-7)*cf*ca^2*nf^2
       + 225728/9*N^(-7)*cf*ca^3*nf
       - 20480/9*N^(-7)*cf^2*nf^3
       - 269264/9*N^(-7)*cf^2*ca*nf^2
       - 194696/9*N^(-7)*cf^2*ca^2*nf
       + 269720/9*N^(-7)*cf^3*nf^2
       + 36844/9*N^(-7)*cf^3*ca*nf
       + 240*N^(-7)*cf^3*beta0^2
       - 36436/3*N^(-7)*cf^4*nf
       + 3200/3*N^(-7)*cf^4*ca
       + 640/3*N^(-7)*cf^4*beta0
       + 1280*N^(-7)*cf^5
       + 1120*N^(-7)*z2*cf*ca^2*nf^2
       + 864*N^(-7)*z2*cf*ca^3*nf
       - 2848*N^(-7)*z2*cf^2*ca*nf^2
       + 19840*N^(-7)*z2*cf^2*ca^2*nf
       - 8192*N^(-7)*z2*cf^3*nf^2
       - 32048*N^(-7)*z2*cf^3*ca*nf
       - 3600*N^(-7)*z2*cf^3*ca^2
       + 26224*N^(-7)*z2*cf^4*nf
       + 11520*N^(-7)*z2*cf^4*ca
       - 11840*N^(-7)*z2*cf^5
      ;

* eq.(5.14)
L   Pqg4R =
       + 7168*N^(-9)*ca^4*nf
       - 10752*N^(-9)*cf*ca^2*nf^2
       - 3584*N^(-9)*cf*ca^3*nf
       + 1792*N^(-9)*cf^2*nf^3
       + 7168*N^(-9)*cf^2*ca*nf^2
       + 1792*N^(-9)*cf^2*ca^2*nf
       - 2688*N^(-9)*cf^3*nf^2
       - 896*N^(-9)*cf^3*ca*nf
       + 448*N^(-9)*cf^4*nf
       + 1792*N^(-8)*ca^3*nf^2
       + 4096/3*N^(-8)*ca^4*nf
       - 1792/3*N^(-8)*cf*ca*nf^3
       + 4736*N^(-8)*cf*ca^2*nf^2
       - 2368*N^(-8)*cf*ca^3*nf
       - 11648/3*N^(-8)*cf^2*nf^3
       - 6272/3*N^(-8)*cf^2*ca*nf^2
       + 7840/3*N^(-8)*cf^2*ca^2*nf
       + 7424/3*N^(-8)*cf^3*nf^2
       - 6064/3*N^(-8)*cf^3*ca*nf
       + 584/3*N^(-8)*cf^4*nf
       + 128*N^(-7)*ca^2*nf^3
       + 83456/27*N^(-7)*ca^3*nf^2
       + 949216/27*N^(-7)*ca^4*nf
       - 128/9*N^(-7)*cf*nf^4
       - 52672/27*N^(-7)*cf*ca*nf^3
       - 1451200/27*N^(-7)*cf*ca^2*nf^2
       - 653008/27*N^(-7)*cf*ca^3*nf
       + 427424/27*N^(-7)*cf^2*nf^3
       + 950504/27*N^(-7)*cf^2*ca*nf^2
       + 293660/27*N^(-7)*cf^2*ca^2*nf
       - 458740/27*N^(-7)*cf^3*nf^2
       - 58586/27*N^(-7)*cf^3*ca*nf
       + 5830/3*N^(-7)*cf^4*nf
       + 2000*N^(-7)*z2*ca^3*nf^2
       + 14576/3*N^(-7)*z2*ca^4*nf
       - 16312/3*N^(-7)*z2*cf*ca^2*nf^2
       + 33368/3*N^(-7)*z2*cf*ca^3*nf
       - 12168*N^(-7)*z2*cf^2*ca*nf^2
       - 17184*N^(-7)*z2*cf^2*ca^2*nf
       + 25696/3*N^(-7)*z2*cf^3*nf^2
       + 54280/3*N^(-7)*z2*cf^3*ca*nf
       - 26432/3*N^(-7)*z2*cf^4*nf
      ;

* eq.(5.15)
L   Pgq4R =
       - 14336*N^(-9)*cf*ca^4
       + 21504*N^(-9)*cf^2*ca^2*nf
       + 7168*N^(-9)*cf^2*ca^3
       - 3584*N^(-9)*cf^3*nf^2
       - 14336*N^(-9)*cf^3*ca*nf
       - 3584*N^(-9)*cf^3*ca^2
       + 5376*N^(-9)*cf^4*nf
       + 1792*N^(-9)*cf^4*ca
       - 896*N^(-9)*cf^5
       - 3584*N^(-8)*cf*ca^3*nf
       - 16128*N^(-8)*cf*ca^4
       + 3584/3*N^(-8)*cf^2*ca*nf^2
       + 46592/3*N^(-8)*cf^2*ca^2*nf
       + 43904/3*N^(-8)*cf^2*ca^3
       + 7168/3*N^(-8)*cf^3*nf^2
       - 51968/3*N^(-8)*cf^3*ca*nf
       - 31808/3*N^(-8)*cf^3*ca^2
       + 4928*N^(-8)*cf^4*nf
       + 6944*N^(-8)*cf^4*ca
       - 2240*N^(-8)*cf^5
       - 256*N^(-7)*cf*ca^2*nf^2
       - 460544/27*N^(-7)*cf*ca^3*nf
       - 1203712/27*N^(-7)*cf*ca^4
       + 256/9*N^(-7)*cf^2*nf^3
       + 318208/27*N^(-7)*cf^2*ca*nf^2
       + 2395936/27*N^(-7)*cf^2*ca^2*nf
       - 21392/27*N^(-7)*cf^2*ca^3
       - 750464/27*N^(-7)*cf^3*nf^2
       - 1016912/27*N^(-7)*cf^3*ca*nf
       + 141688/27*N^(-7)*cf^3*ca^2
       + 512320/27*N^(-7)*cf^4*nf
       - 106504/27*N^(-7)*cf^4*ca
       - 1568*N^(-7)*cf^5
       - 4000*N^(-7)*z2*cf*ca^3*nf
       - 60512/3*N^(-7)*z2*cf*ca^4
       + 56944/3*N^(-7)*z2*cf^2*ca^2*nf
       - 12656/3*N^(-7)*z2*cf^2*ca^3
       + 12176*N^(-7)*z2*cf^3*ca*nf
       + 23008*N^(-7)*z2*cf^3*ca^2
       - 39232/3*N^(-7)*z2*cf^4*nf
       - 93280/3*N^(-7)*z2*cf^4*ca
       + 48944/3*N^(-7)*z2*cf^5
      ;

* eq.(5.16)
L   Pgg4R =
       - 14336*N^(-9)*ca^5
       + 28672*N^(-9)*cf*ca^3*nf
       - 10752*N^(-9)*cf^2*ca*nf^2
       - 10752*N^(-9)*cf^2*ca^2*nf
       + 3584*N^(-9)*cf^3*nf^2
       + 3584*N^(-9)*cf^3*ca*nf
       - 896*N^(-9)*cf^4*nf
       - 17920/3*N^(-8)*ca^4*nf
       - 8960/3*N^(-8)*ca^5
       + 5376*N^(-8)*cf*ca^2*nf^2
       - 14848/3*N^(-8)*cf*ca^3*nf
       - 896/3*N^(-8)*cf^2*nf^3
       + 10048*N^(-8)*cf^2*ca*nf^2
       - 15040/3*N^(-8)*cf^2*ca^2*nf
       - 11264/3*N^(-8)*cf^3*nf^2
       + 9536/3*N^(-8)*cf^3*ca*nf
       - 1168/3*N^(-8)*cf^4*nf
       - 2560/3*N^(-7)*ca^3*nf^2
       - 104960/9*N^(-7)*ca^4*nf
       - 580480/9*N^(-7)*ca^5
       + 256*N^(-7)*cf*ca*nf^3
       + 104768/9*N^(-7)*cf*ca^2*nf^2
       + 1218112/9*N^(-7)*cf*ca^3*nf
       - 14080/9*N^(-7)*cf^2*nf^3
       - 613936/9*N^(-7)*cf^2*ca*nf^2
       - 347704/9*N^(-7)*cf^2*ca^2*nf
       + 212840/9*N^(-7)*cf^3*nf^2
       + 73396/9*N^(-7)*cf^3*ca*nf
       - 8684/3*N^(-7)*cf^4*nf
       - 5760*N^(-7)*z2*ca^4*nf
       - 28160*N^(-7)*z2*ca^5
       + 1760*N^(-7)*z2*cf*ca^2*nf^2
       + 35616*N^(-7)*z2*cf*ca^3*nf
       - 7072*N^(-7)*z2*cf^2*ca*nf^2
       + 13280*N^(-7)*z2*cf^2*ca^2*nf
       - 4608*N^(-7)*z2*cf^3*nf^2
       - 17392*N^(-7)*z2*cf^3*ca*nf
       + 10256*N^(-7)*z2*cf^4*nf
      ;

* eq.(5.24)
L   C2q4R =
       - 3120*N^(-8)*cf*ca^2*nf
       + 1560*N^(-8)*cf^2*nf^2
       + 3120*N^(-8)*cf^2*ca*nf
       - 2340*N^(-8)*cf^3*nf
       + 390*N^(-8)*cf^4
       + 5216/9*N^(-7)*cf*ca*nf^2
       - 60872/9*N^(-7)*cf*ca^2*nf
       - 16688/9*N^(-7)*cf^2*nf^2
       + 86228/9*N^(-7)*cf^2*ca*nf
       - 7798/3*N^(-7)*cf^3*nf
       - 1822/3*N^(-7)*cf^3*beta0
       + 1052*N^(-7)*cf^4
       - 952/9*N^(-6)*cf*nf^3
       + 9848/27*N^(-6)*cf*ca*nf^2
       - 485222/27*N^(-6)*cf*ca^2*nf
       + 248786/27*N^(-6)*cf^2*nf^2
       + 194008/27*N^(-6)*cf^2*ca*nf
       + 1951/6*N^(-6)*cf^2*beta0^2
       - 5537*N^(-6)*cf^3*nf
       + 2180/3*N^(-6)*cf^3*ca
       - 448*N^(-6)*cf^3*beta0
       + 336*N^(-6)*cf^4
       + 5056/3*N^(-6)*z2*cf*ca^2*nf
       - 1088*N^(-6)*z2*cf^2*nf^2
       - 6056*N^(-6)*z2*cf^2*ca*nf
       - 1560*N^(-6)*z2*cf^2*ca^2
       + 21752/3*N^(-6)*z2*cf^3*nf
       + 4992*N^(-6)*z2*cf^3*ca
       - 5872*N^(-6)*z2*cf^4
      ;

* eq.(5.25)
L   C2g4R =
       - 3120*N^(-8)*ca^3*nf
       + 3120*N^(-8)*cf*ca*nf^2
       + 1560*N^(-8)*cf*ca^2*nf
       - 1560*N^(-8)*cf^2*nf^2
       - 780*N^(-8)*cf^2*ca*nf
       + 390*N^(-8)*cf^3*nf
       + 536/9*N^(-7)*ca^2*nf^2
       - 35132/9*N^(-7)*ca^3*nf
       - 2608/9*N^(-7)*cf*nf^3
       - 2056/3*N^(-7)*cf*ca*nf^2
       + 30052/9*N^(-7)*cf*ca^2*nf
       + 13778/9*N^(-7)*cf^2*nf^2
       - 21101/9*N^(-7)*cf^2*ca*nf
       + 889/3*N^(-7)*cf^3*nf
       - 248/27*N^(-6)*ca*nf^3
       - 11528/27*N^(-6)*ca^2*nf^2
       - 489110/27*N^(-6)*ca^3*nf
       + 20300/27*N^(-6)*cf*nf^3
       + 13364*N^(-6)*cf*ca*nf^2
       + 218755/27*N^(-6)*cf*ca^2*nf
       - 204632/27*N^(-6)*cf^2*nf^2
       - 19957/27*N^(-6)*cf^2*ca*nf
       + 2453/6*N^(-6)*cf^3*nf
       - 244*N^(-6)*z2*ca^2*nf^2
       - 940/3*N^(-6)*z2*ca^3*nf
       - 2132/3*N^(-6)*z2*cf*ca*nf^2
       - 3752*N^(-6)*z2*cf*ca^2*nf
       + 8864/3*N^(-6)*z2*cf^2*nf^2
       + 16160/3*N^(-6)*z2*cf^2*ca*nf
       - 11908/3*N^(-6)*z2*cf^3*nf
      ;

* eq.(5.26)
L   CLq4R =
       - 1920*N^(-6)*cf*ca^2*nf
       + 960*N^(-6)*cf^2*nf^2
       + 1920*N^(-6)*cf^2*ca*nf
       - 1440*N^(-6)*cf^3*nf
       + 240*N^(-6)*cf^4
       + 2176/9*N^(-5)*cf*ca*nf^2
       - 24640/9*N^(-5)*cf*ca^2*nf
       - 13024/9*N^(-5)*cf^2*nf^2
       + 37408/9*N^(-5)*cf^2*ca*nf
       - 2048/3*N^(-5)*cf^3*nf
       - 992/3*N^(-5)*cf^3*beta0
       + 472*N^(-5)*cf^4
       - 128/3*N^(-4)*cf*nf^3
       - 5696/27*N^(-4)*cf*ca*nf^2
       - 156352/27*N^(-4)*cf*ca^2*nf
       + 135376/27*N^(-4)*cf^2*nf^2
       - 39304/27*N^(-4)*cf^2*ca*nf
       + 460/3*N^(-4)*cf^2*beta0^2
       - 1568/3*N^(-4)*cf^3*nf
       + 480*N^(-4)*cf^3*ca
       + 56*N^(-4)*cf^3*beta0
       - 644*N^(-4)*cf^4
       + 5152/3*N^(-4)*z2*cf*ca^2*nf
       - 704*N^(-4)*z2*cf^2*nf^2
       - 3552*N^(-4)*z2*cf^2*ca*nf
       - 1200*N^(-4)*z2*cf^2*ca^2
       + 11552/3*N^(-4)*z2*cf^3*nf
       + 3840*N^(-4)*z2*cf^3*ca
       - 4016*N^(-4)*z2*cf^4
      ;

* eq.(5.27)
L   CLg4R =
       - 1920*N^(-6)*ca^3*nf
       + 1920*N^(-6)*cf*ca*nf^2
       + 960*N^(-6)*cf*ca^2*nf
       - 960*N^(-6)*cf^2*nf^2
       - 480*N^(-6)*cf^2*ca*nf
       + 240*N^(-6)*cf^3*nf
       - 704/9*N^(-5)*ca^2*nf^2
       - 8800/9*N^(-5)*ca^3*nf
       - 1088/9*N^(-5)*cf*nf^3
       - 4640/3*N^(-5)*cf*ca*nf^2
       + 9248/9*N^(-5)*cf*ca^2*nf
       + 13648/9*N^(-5)*cf^2*nf^2
       - 8296/9*N^(-5)*cf^2*ca*nf
       - 16/3*N^(-5)*cf^3*nf
       - 64/27*N^(-4)*ca*nf^3
       - 8416/27*N^(-4)*ca^2*nf^2
       - 202048/27*N^(-4)*ca^3*nf
       + 11776/27*N^(-4)*cf*nf^3
       + 19696/3*N^(-4)*cf*ca*nf^2
       + 70280/27*N^(-4)*cf*ca^2*nf
       - 113992/27*N^(-4)*cf^2*nf^2
       + 25948/27*N^(-4)*cf^2*ca*nf
       - 460/3*N^(-4)*cf^3*nf
       - 192*N^(-4)*z2*ca^2*nf^2
       + 544*N^(-4)*z2*ca^3*nf
       - 1888/3*N^(-4)*z2*cf*ca*nf^2
       - 2288*N^(-4)*z2*cf*ca^2*nf
       + 4672/3*N^(-4)*z2*cf^2*nf^2
       + 10688/3*N^(-4)*z2*cf^2*ca*nf
       - 7856/3*N^(-4)*z2*cf^3*nf
      ;

* eq.(5.28)
L   C2q5R =
       + 42432*N^(-10)*cf*ca^3*nf
       - 42432*N^(-10)*cf^2*ca*nf^2
       - 42432*N^(-10)*cf^2*ca^2*nf
       + 31824*N^(-10)*cf^3*nf^2
       + 31824*N^(-10)*cf^3*ca*nf
       - 21216*N^(-10)*cf^4*nf
       + 2652*N^(-10)*cf^5
       - 81248/9*N^(-9)*cf*ca^2*nf^2
       + 5366608/45*N^(-9)*cf*ca^3*nf
       + 72448/9*N^(-9)*cf^2*nf^3
       - 1361056/45*N^(-9)*cf^2*ca*nf^2
       - 7102528/45*N^(-9)*cf^2*ca^2*nf
       - 243376/15*N^(-9)*cf^3*nf^2
       + 2208812/15*N^(-9)*cf^3*ca*nf
       - 511648/15*N^(-9)*cf^4*nf
       - 17012/3*N^(-9)*cf^4*beta0
       + 8418*N^(-9)*cf^5
       + 17696/9*N^(-8)*cf*ca*nf^3
       - 1647376/135*N^(-8)*cf*ca^2*nf^2
       + 49019344/135*N^(-8)*cf*ca^3*nf
       - 2472352/135*N^(-8)*cf^2*nf^3
       - 1373120/9*N^(-8)*cf^2*ca*nf^2
       - 50030192/135*N^(-8)*cf^2*ca^2*nf
       + 22243424/135*N^(-8)*cf^3*nf^2
       + 2986142/27*N^(-8)*cf^3*ca*nf
       + 14363/3*N^(-8)*cf^3*beta0^2
       - 298372/5*N^(-8)*cf^4*nf
       + 6040*N^(-8)*cf^4*ca
       - 23546/3*N^(-8)*cf^4*beta0
       + 6438*N^(-8)*cf^5
       + 4400*N^(-8)*z2*cf*ca^2*nf^2
       - 40912/5*N^(-8)*z2*cf*ca^3*nf
       + 160112/15*N^(-8)*z2*cf^2*ca*nf^2
       + 328736/3*N^(-8)*z2*cf^2*ca^2*nf
       - 930992/15*N^(-8)*z2*cf^3*nf^2
       - 2574688/15*N^(-8)*z2*cf^3*ca*nf
       - 15840*N^(-8)*z2*cf^3*ca^2
       + 721568/5*N^(-8)*z2*cf^4*nf
       + 50688*N^(-8)*z2*cf^4*ca
       - 56508*N^(-8)*z2*cf^5
      ;

* eq.(5.29)
L   C2g5R =
       + 42432*N^(-10)*ca^4*nf
       - 63648*N^(-10)*cf*ca^2*nf^2
       - 21216*N^(-10)*cf*ca^3*nf
       + 10608*N^(-10)*cf^2*nf^3
       + 42432*N^(-10)*cf^2*ca*nf^2
       + 10608*N^(-10)*cf^2*ca^2*nf
       - 15912*N^(-10)*cf^3*nf^2
       - 5304*N^(-10)*cf^3*ca*nf
       + 2652*N^(-10)*cf^4*nf
       - 17600/9*N^(-9)*ca^3*nf^2
       + 3616288/45*N^(-9)*ca^4*nf
       + 81248/9*N^(-9)*cf*ca*nf^3
       - 1868624/45*N^(-9)*cf*ca^2*nf^2
       - 2668904/45*N^(-9)*cf*ca^3*nf
       - 1239104/45*N^(-9)*cf^2*nf^3
       + 1566016/45*N^(-9)*cf^2*ca*nf^2
       + 1762432/45*N^(-9)*cf^2*ca^2*nf
       + 334184/45*N^(-9)*cf^3*nf^2
       - 1089206/45*N^(-9)*cf^3*ca*nf
       + 50356/15*N^(-9)*cf^4*nf
       + 12464/27*N^(-8)*ca^2*nf^3
       + 636112/135*N^(-8)*ca^3*nf^2
       + 45401116/135*N^(-8)*ca^4*nf
       - 8848/9*N^(-8)*cf*nf^4
       - 1105856/135*N^(-8)*cf*ca*nf^3
       - 48126868/135*N^(-8)*cf*ca^2*nf^2
       - 2593172/15*N^(-8)*cf*ca^3*nf
       + 14000444/135*N^(-8)*cf^2*nf^3
       + 3852416/27*N^(-8)*cf^2*ca*nf^2
       + 3955174/45*N^(-8)*cf^2*ca^2*nf
       - 1852186/27*N^(-8)*cf^3*nf^2
       - 1463644/135*N^(-8)*cf^3*ca*nf
       + 59357/15*N^(-8)*cf^4*nf
       + 7232*N^(-8)*z2*ca^3*nf^2
       + 59264/3*N^(-8)*z2*ca^4*nf
       - 1112/3*N^(-8)*z2*cf*ca^2*nf^2
       + 232664/5*N^(-8)*z2*cf*ca^3*nf
       - 7152*N^(-8)*z2*cf^2*nf^3
       - 1116824/15*N^(-8)*z2*cf^2*ca*nf^2
       - 262160/3*N^(-8)*z2*cf^2*ca^2*nf
       + 255768/5*N^(-8)*z2*cf^3*nf^2
       + 1424656/15*N^(-8)*z2*cf^3*ca*nf
       - 673108/15*N^(-8)*z2*cf^4*nf
      ;

* eq.(5.30)
L   CLq5R =
       + 24960*N^(-8)*cf*ca^3*nf
       - 24960*N^(-8)*cf^2*ca*nf^2
       - 24960*N^(-8)*cf^2*ca^2*nf
       + 18720*N^(-8)*cf^3*nf^2
       + 18720*N^(-8)*cf^3*ca*nf
       - 12480*N^(-8)*cf^4*nf
       + 1560*N^(-8)*cf^5
       - 30400/9*N^(-7)*cf*ca^2*nf^2
       + 436192/9*N^(-7)*cf*ca^3*nf
       + 33920/9*N^(-7)*cf^2*nf^3
       - 27328/9*N^(-7)*cf^2*ca*nf^2
       - 602368/9*N^(-7)*cf^2*ca^2*nf
       - 56272/3*N^(-7)*cf^3*nf^2
       + 198248/3*N^(-7)*cf^3*ca*nf
       - 33904/3*N^(-7)*cf^4*nf
       - 8920/3*N^(-7)*cf^4*beta0
       + 3736*N^(-7)*cf^5
       + 6208/9*N^(-6)*cf*ca*nf^3
       + 44480/27*N^(-6)*cf*ca^2*nf^2
       + 3501232/27*N^(-6)*cf*ca^3*nf
       - 305536/27*N^(-6)*cf^2*nf^3
       - 594880/9*N^(-6)*cf^2*ca*nf^2
       - 2883632/27*N^(-6)*cf^2*ca^2*nf
       + 2227784/27*N^(-6)*cf^3*nf^2
       - 417656/27*N^(-6)*cf^3*ca*nf
       + 6574/3*N^(-6)*cf^3*beta0^2
       - 20104/3*N^(-6)*cf^4*nf
       + 11120/3*N^(-6)*cf^4*ca
       - 1504*N^(-6)*cf^4*beta0
       - 2064*N^(-6)*cf^5
       + 3040*N^(-6)*z2*cf*ca^2*nf^2
       - 36640/3*N^(-6)*z2*cf*ca^3*nf
       + 24224/3*N^(-6)*z2*cf^2*ca*nf^2
       + 65152*N^(-6)*z2*cf^2*ca^2*nf
       - 98048/3*N^(-6)*z2*cf^3*nf^2
       - 101216*N^(-6)*z2*cf^3*ca*nf
       - 10800*N^(-6)*z2*cf^3*ca^2
       + 248416/3*N^(-6)*z2*cf^4*nf
       + 34560*N^(-6)*z2*cf^4*ca
       - 35648*N^(-6)*z2*cf^5
      ;

* eq.(5.31)
L   CLg5R =
       + 24960*N^(-8)*ca^4*nf
       - 37440*N^(-8)*cf*ca^2*nf^2
       - 12480*N^(-8)*cf*ca^3*nf
       + 6240*N^(-8)*cf^2*nf^3
       + 24960*N^(-8)*cf^2*ca*nf^2
       + 6240*N^(-8)*cf^2*ca^2*nf
       - 9360*N^(-8)*cf^3*nf^2
       - 3120*N^(-8)*cf^3*ca*nf
       + 1560*N^(-8)*cf^4*nf
       + 7040/9*N^(-7)*ca^3*nf^2
       + 230272/9*N^(-7)*ca^4*nf
       + 30400/9*N^(-7)*cf*ca*nf^3
       + 22528/9*N^(-7)*cf*ca^2*nf^2
       - 178352/9*N^(-7)*cf*ca^3*nf
       - 164144/9*N^(-7)*cf^2*nf^3
       - 1856/9*N^(-7)*cf^2*ca*nf^2
       + 139120/9*N^(-7)*cf^2*ca^2*nf
       + 96152/9*N^(-7)*cf^3*nf^2
       - 93860/9*N^(-7)*cf^3*ca*nf
       + 2140/3*N^(-7)*cf^4*nf
       + 3424/27*N^(-6)*ca^2*nf^3
       + 158944/27*N^(-6)*ca^3*nf^2
       + 3671992/27*N^(-6)*ca^4*nf
       - 3104/9*N^(-6)*cf*nf^4
       - 200672/27*N^(-6)*cf*ca*nf^3
       - 4337080/27*N^(-6)*cf*ca^2*nf^2
       - 180424/3*N^(-6)*cf*ca^3*nf
       + 1630352/27*N^(-6)*cf^2*nf^3
       + 1532152/27*N^(-6)*cf^2*ca*nf^2
       + 23620*N^(-6)*cf^2*ca^2*nf
       - 923060/27*N^(-6)*cf^3*nf^2
       + 191024/27*N^(-6)*cf^3*ca*nf
       - 2522/3*N^(-6)*cf^4*nf
       + 4992*N^(-6)*z2*ca^3*nf^2
       + 3776*N^(-6)*z2*ca^4*nf
       + 16304/3*N^(-6)*z2*cf*ca^2*nf^2
       + 74672/3*N^(-6)*z2*cf*ca^3*nf
       - 4352*N^(-6)*z2*cf^2*nf^3
       - 115664/3*N^(-6)*z2*cf^2*ca*nf^2
       - 53616*N^(-6)*z2*cf^2*ca^2*nf
       + 26080*N^(-6)*z2*cf^3*nf^2
       + 181376/3*N^(-6)*z2*cf^3*ca*nf
       - 84112/3*N^(-6)*z2*cf^4*nf
      ;

* -----------------------------------------------------------------------------
*
* Appendix A
*
* -----------------------------------------------------------------------------

* eq.(A.1)
L   Pns0R =
       + 2*N^(-1)*cf
       + cf
       + 2*N*cf
       - 4*N*z2*cf
       - 2*N^2*cf
       + 4*N^2*z3*cf
      ;

L   Pns1R =
       + 4*N^(-3)*cf^2
       + 4/3*N^(-2)*cf*nf
       - 22/3*N^(-2)*cf*ca
       + 4*N^(-2)*cf^2
       - 44/9*N^(-1)*cf*nf
       + 302/9*N^(-1)*cf*ca
       - 4*N^(-1)*cf^2
       - 8*N^(-1)*z2*cf^2
       + 29/9*cf*nf
       - 421/18*cf*ca
       + 19/2*cf^2
       - 12*z3*cf*ca
       + 16*z3*cf^2
      ;

L   Pns2R =
       + 16*N^(-5)*cf^3
       + 8*N^(-4)*cf^2*nf
       - 44*N^(-4)*cf^2*ca
       + 24*N^(-4)*cf^3
       + 8/9*N^(-3)*cf*nf^2
       - 88/9*N^(-3)*cf*ca*nf
       + 242/9*N^(-3)*cf*ca^2
       - 128/9*N^(-3)*cf^2*nf
       + 944/9*N^(-3)*cf^2*ca
       + 8*N^(-3)*cf^3
       - 60*N^(-3)*z2*cf*ca^2
       + 192*N^(-3)*z2*cf^2*ca
       - 208*N^(-3)*z2*cf^3
       - 88/27*N^(-2)*cf*nf^2
       + 1268/27*N^(-2)*cf*ca*nf
       - 3934/27*N^(-2)*cf*ca^2
       + 88/9*N^(-2)*cf^2*nf
       - 370/9*N^(-2)*cf^2*ca
       + 30*N^(-2)*cf^3
       - 48*N^(-2)*z3*cf^2*ca
       + 96*N^(-2)*z3*cf^3
       - 8*N^(-2)*z2*cf*ca*nf
       + 92*N^(-2)*z2*cf*ca^2
       - 216*N^(-2)*z2*cf^2*ca
       + 192*N^(-2)*z2*cf^3
      ;

* eq.(A.2)
L   C2ns1R =
       + 2*N^(-2)*cf
       + 3*N^(-1)*cf
       - 5*cf
       - 2*z2*cf
       - 4*N*cf
       - 2*N*z3*cf
       + 5*N*z2*cf
      ;

L   C2ns1e1R =
       - 2*N^(-3)*cf
       - 3*N^(-2)*cf
       + 5*N^(-1)*cf
       + 3*N^(-1)*z2*cf
       - 10*cf
       + 7/2*z2*cf
      ;

L   C2ns1e2R =
       + 2*N^(-4)*cf
       + 3*N^(-3)*cf
       - 5*N^(-2)*cf
       - 3*N^(-2)*z2*cf
       + 10*N^(-1)*cf
       + 14/3*N^(-1)*z3*cf
       - 9/2*N^(-1)*z2*cf
      ;

L   C2ns2R =
       + 10*N^(-4)*cf^2
       + 10/3*N^(-3)*cf*nf
       - 55/3*N^(-3)*cf*ca
       + 18*N^(-3)*cf^2
       - 4*N^(-2)*cf*nf
       + 32*N^(-2)*cf*ca
       - 17*N^(-2)*cf^2
       - 24*N^(-2)*z2*cf^2
       + 89/27*N^(-1)*cf*nf
       - 1693/54*N^(-1)*cf*ca
       + 3/2*N^(-1)*cf^2
       - 12*N^(-1)*z3*cf*ca
       + 56*N^(-1)*z3*cf^2
       - 8/3*N^(-1)*z2*cf*nf
       + 44/3*N^(-1)*z2*cf*ca
       - 8*N^(-1)*z2*cf^2
      ;

L   C2ns2e1R =
       - 26*N^(-5)*cf^2
       - 26/3*N^(-4)*cf*nf
       + 143/3*N^(-4)*cf*ca
       - 50*N^(-4)*cf^2
       + 64/9*N^(-3)*cf*nf
       - 562/9*N^(-3)*cf*ca
       + 47*N^(-3)*cf^2
       + 68*N^(-3)*z2*cf^2
       - 20/3*N^(-2)*cf*nf
       + 212/3*N^(-2)*cf*ca
       - 49*N^(-2)*cf^2
       + 24*N^(-2)*z3*cf*ca
       - 128*N^(-2)*z3*cf^2
       + 28/3*N^(-2)*z2*cf*nf
       - 154/3*N^(-2)*z2*cf*ca
       + 54*N^(-2)*z2*cf^2
      ;

L   C2ns3R =
       + 60*N^(-6)*cf^3
       + 364/9*N^(-5)*cf^2*nf
       - 2002/9*N^(-5)*cf^2*ca
       + 134*N^(-5)*cf^3
       + 184/27*N^(-4)*cf*nf^2
       - 2024/27*N^(-4)*cf*ca*nf
       + 5566/27*N^(-4)*cf*ca^2
       - 10/9*N^(-4)*cf^2*nf
       + 835/9*N^(-4)*cf^2*ca
       - 30*N^(-4)*cf^3
       - 120*N^(-4)*z2*cf*ca^2
       + 384*N^(-4)*z2*cf^2*ca
       - 524*N^(-4)*z2*cf^3
       - 992/81*N^(-3)*cf*nf^2
       + 14392/81*N^(-3)*cf*ca*nf
       - 46124/81*N^(-3)*cf*ca^2
       - 2630/81*N^(-3)*cf^2*nf
       + 17315/81*N^(-3)*cf^2*ca
       - 113/3*N^(-3)*cf^3
       - 128*N^(-3)*z3*cf^2*ca
       + 1292/3*N^(-3)*z3*cf^3
       - 16*N^(-3)*z2*cf*ca*nf
       + 168*N^(-3)*z2*cf*ca^2
       - 532/9*N^(-3)*z2*cf^2*nf
       - 530/9*N^(-3)*z2*cf^2*ca
       + 598/3*N^(-3)*z2*cf^3
      ;

* eq.(A.3)
L   CLns1R =
       + 4*cf
       - 4*N*cf
       + 4*N^2*cf
      ;

L   CLns1e1R =
       + 4*cf
       - 4*N*cf
       + 4*N*z2*cf
      ;

L   CLns1e2R =
       + 8*cf
       - 2*z2*cf
      ;

L   CLns2R =
       + 8*N^(-2)*cf^2
       + 8/3*N^(-1)*cf*nf
       - 44/3*N^(-1)*cf*ca
       + 12*N^(-1)*cf^2
       - 76/9*cf*nf
       + 538/9*cf*ca
       - 74*cf^2
       - 8*z2*cf^2
      ;

L   CLns2e1R =
       - 8*N^(-3)*cf^2
       - 8/3*N^(-2)*cf*nf
       + 44/3*N^(-2)*cf*ca
       - 4*N^(-2)*cf^2
       + 100/9*N^(-1)*cf*nf
       - 670/9*N^(-1)*cf*ca
       + 70*N^(-1)*cf^2
       + 20*N^(-1)*z2*cf^2
      ;

L   CLns3R =
       + 40*N^(-4)*cf^3
       + 24*N^(-3)*cf^2*nf
       - 132*N^(-3)*cf^2*ca
       + 64*N^(-3)*cf^3
       + 32/9*N^(-2)*cf*nf^2
       - 352/9*N^(-2)*cf*ca*nf
       + 968/9*N^(-2)*cf*ca^2
       - 224/9*N^(-2)*cf^2*nf
       + 1832/9*N^(-2)*cf^2*ca
       - 168*N^(-2)*cf^3
       - 120*N^(-2)*z2*cf*ca^2
       + 384*N^(-2)*z2*cf^2*ca
       - 416*N^(-2)*z2*cf^3
      ;

* eq.(A.4)
L   C3ns1R =
       + 2*N^(-2)*cf
       + N^(-1)*cf
       - 7*cf
       - 2*z2*cf
       - 2*N*cf
       - 2*N*z3*cf
       + 5*N*z2*cf
      ;

L   C3ns1e1R =
       - 2*N^(-3)*cf
       - N^(-2)*cf
       + N^(-1)*cf
       + 3*N^(-1)*z2*cf
       - 14*cf
       + 3/2*z2*cf
      ;

L   C3ns1e2R =
       + 2*N^(-4)*cf
       + N^(-3)*cf
       - N^(-2)*cf
       - 3*N^(-2)*z2*cf
       + 14/3*N^(-1)*z3*cf
       - 3/2*N^(-1)*z2*cf
      ;

L   C3ns2R =
       + 10*N^(-4)*cf^2
       + 10/3*N^(-3)*cf*nf
       - 55/3*N^(-3)*cf*ca
       + 10*N^(-3)*cf^2
       - 20/3*N^(-2)*cf*nf
       + 140/3*N^(-2)*cf*ca
       - 33*N^(-2)*cf^2
       - 24*N^(-2)*z2*cf^2
       + 131/27*N^(-1)*cf*nf
       - 2515/54*N^(-1)*cf*ca
       + 29/2*N^(-1)*cf^2
       - 12*N^(-1)*z3*cf*ca
       + 56*N^(-1)*z3*cf^2
       - 8/3*N^(-1)*z2*cf*nf
       + 44/3*N^(-1)*z2*cf*ca
       + 4*N^(-1)*z2*cf^2
      ;

L   C3ns2e1R =
       - 26*N^(-5)*cf^2
       - 26/3*N^(-4)*cf*nf
       + 143/3*N^(-4)*cf*ca
       - 26*N^(-4)*cf^2
       + 136/9*N^(-3)*cf*nf
       - 958/9*N^(-3)*cf*ca
       + 71*N^(-3)*cf^2
       + 68*N^(-3)*z2*cf^2
       - 50/3*N^(-2)*cf*nf
       + 437/3*N^(-2)*cf*ca
       - 120*N^(-2)*cf^2
       + 24*N^(-2)*z3*cf*ca
       - 128*N^(-2)*z3*cf^2
       + 28/3*N^(-2)*z2*cf*nf
       - 154/3*N^(-2)*z2*cf*ca
       + 8*N^(-2)*z2*cf^2
      ;

L   C3ns3R =
       + 60*N^(-6)*cf^3
       + 364/9*N^(-5)*cf^2*nf
       - 2002/9*N^(-5)*cf^2*ca
       + 90*N^(-5)*cf^3
       + 184/27*N^(-4)*cf*nf^2
       - 2024/27*N^(-4)*cf*ca*nf
       + 5566/27*N^(-4)*cf*ca^2
       - 286/9*N^(-4)*cf^2*nf
       + 2353/9*N^(-4)*cf^2*ca
       - 142*N^(-4)*cf^3
       - 120*N^(-4)*z2*cf*ca^2
       + 384*N^(-4)*z2*cf^2*ca
       - 524*N^(-4)*z2*cf^3
       - 1424/81*N^(-3)*cf*nf^2
       + 19144/81*N^(-3)*cf*ca*nf
       - 59192/81*N^(-3)*cf*ca^2
       - 3818/81*N^(-3)*cf^2*nf
       + 18989/81*N^(-3)*cf^2*ca
       - 47/3*N^(-3)*cf^3
       - 128*N^(-3)*z3*cf^2*ca
       + 1292/3*N^(-3)*z3*cf^3
       - 16*N^(-3)*z2*cf*ca*nf
       + 228*N^(-3)*z2*cf*ca^2
       - 532/9*N^(-3)*z2*cf^2*nf
       - 2258/9*N^(-3)*z2*cf^2*ca
       + 1438/3*N^(-3)*z2*cf^3
      ;

* eq.(A.5)
L   Pqq0R =
       + 2*N^(-1)*cf
       + cf
       + 2*N*cf
       - 4*N*z2*cf
      ;

L   Pqq1R =
       - 8*N^(-3)*cf*nf
       + 4*N^(-3)*cf^2
       - 8/3*N^(-2)*cf*nf
       - 22/3*N^(-2)*cf*ca
       + 4*N^(-2)*cf^2
       - 116/9*N^(-1)*cf*nf
       + 302/9*N^(-1)*cf*ca
       - 4*N^(-1)*cf^2
       - 8*N^(-1)*z2*cf^2
      ;

L   Pqq2R =
       + 64*N^(-5)*cf*ca*nf
       - 64*N^(-5)*cf^2*nf
       + 16*N^(-5)*cf^3
       - 16/3*N^(-4)*cf*nf^2
       + 232/3*N^(-4)*cf*ca*nf
       - 16*N^(-4)*cf^2*nf
       - 44*N^(-4)*cf^2*ca
       + 24*N^(-4)*cf^3
       + 80/3*N^(-3)*cf*nf^2
       + 316/9*N^(-3)*cf*ca*nf
       + 242/9*N^(-3)*cf*ca^2
       - 1568/9*N^(-3)*cf^2*nf
       + 944/9*N^(-3)*cf^2*ca
       + 8*N^(-3)*cf^3
       + 8*N^(-3)*z2*cf*ca*nf
       - 60*N^(-3)*z2*cf*ca^2
       + 96*N^(-3)*z2*cf^2*nf
       + 192*N^(-3)*z2*cf^2*ca
       - 208*N^(-3)*z2*cf^3
      ;

L   Pqg0R =
       + 2*N^(-1)*nf
       - 2*nf
       + 3*N*nf
      ;

L   Pqg1R =
       - 8*N^(-3)*ca*nf
       + 4*N^(-3)*cf*nf
       - 4*N^(-2)*ca*nf
       - 6*N^(-2)*cf*nf
       - 8*N^(-1)*ca*nf
       + 28*N^(-1)*cf*nf
       - 8*N^(-1)*z2*cf*nf
      ;

L   Pqg2R =
       + 64*N^(-5)*ca^2*nf
       - 32*N^(-5)*cf*nf^2
       - 32*N^(-5)*cf*ca*nf
       + 16*N^(-5)*cf^2*nf
       + 16/3*N^(-4)*ca*nf^2
       + 56/3*N^(-4)*ca^2*nf
       + 152/3*N^(-4)*cf*nf^2
       - 44/3*N^(-4)*cf*ca*nf
       - 12*N^(-4)*cf^2*nf
       + 64/9*N^(-3)*ca*nf^2
       + 1724/9*N^(-3)*ca^2*nf
       - 1370/9*N^(-3)*cf*nf^2
       - 1171/9*N^(-3)*cf*ca*nf
       + 89*N^(-3)*cf^2*nf
       - 12*N^(-3)*z2*ca^2*nf
       + 96*N^(-3)*z2*cf*ca*nf
       - 56*N^(-3)*z2*cf^2*nf
      ;

L   Pgq0R =
       - 4*N^(-1)*cf
       - 2*cf
       - 6*N*cf
      ;

L   Pgq1R =
       + 16*N^(-3)*cf*ca
       - 8*N^(-3)*cf^2
       + 16*N^(-2)*cf*ca
       - 8*N^(-2)*cf^2
       + 128/9*N^(-1)*cf*nf
       - 332/9*N^(-1)*cf*ca
       + 14*N^(-1)*cf^2
       + 16*N^(-1)*z2*cf*ca
      ;

L   Pgq2R =
       - 128*N^(-5)*cf*ca^2
       + 64*N^(-5)*cf^2*nf
       + 64*N^(-5)*cf^2*ca
       - 32*N^(-5)*cf^3
       - 32/3*N^(-4)*cf*ca*nf
       - 400/3*N^(-4)*cf*ca^2
       - 16/3*N^(-4)*cf^2*nf
       + 376/3*N^(-4)*cf^2*ca
       - 48*N^(-4)*cf^3
       - 992/9*N^(-3)*cf*ca*nf
       - 280/9*N^(-3)*cf*ca^2
       + 2380/9*N^(-3)*cf^2*nf
       - 2446/9*N^(-3)*cf^2*ca
       + 42*N^(-3)*cf^3
       - 104*N^(-3)*z2*cf*ca^2
       + 48*N^(-3)*z2*cf^3
      ;

L   Pgg0R =
       - 4*N^(-1)*ca
       - 2/3*nf
       + 5/3*ca
       - 7*N*ca
       - 4*N*z2*ca
      ;

L   Pgg1R =
       + 16*N^(-3)*ca^2
       - 8*N^(-3)*cf*nf
       + 8/3*N^(-2)*ca*nf
       + 4/3*N^(-2)*ca^2
       + 12*N^(-2)*cf*nf
       + 76/9*N^(-1)*ca*nf
       + 74/9*N^(-1)*ca^2
       - 32*N^(-1)*cf*nf
       + 16*N^(-1)*z2*ca^2
      ;

L   Pgg2R =
       - 128*N^(-5)*ca^3
       + 128*N^(-5)*cf*ca*nf
       - 32*N^(-5)*cf^2*nf
       - 32*N^(-4)*ca^2*nf
       - 16*N^(-4)*ca^3
       + 16/3*N^(-4)*cf*nf^2
       - 232/3*N^(-4)*cf*ca*nf
       + 24*N^(-4)*cf^2*nf
       - 16/9*N^(-3)*ca*nf^2
       - 208/3*N^(-3)*ca^2*nf
       - 2612/9*N^(-3)*ca^3
       + 184/9*N^(-3)*cf*nf^2
       + 3548/9*N^(-3)*cf*ca*nf
       - 120*N^(-3)*cf^2*nf
       - 24*N^(-3)*z2*ca^2*nf
       - 160*N^(-3)*z2*ca^3
       + 96*N^(-3)*z2*cf*ca*nf
       + 32*N^(-3)*z2*cf^2*nf
      ;

* eq.(A.6)
L   C2q1R =
       + 2*N^(-2)*cf
       + 3*N^(-1)*cf
       - 5*cf
       - 2*z2*cf
      ;

L   C2q1e1R =
       - 2*N^(-3)*cf
       - 3*N^(-2)*cf
       + 5*N^(-1)*cf
       + 3*N^(-1)*z2*cf
      ;

L   C2q1e2R =
       + 2*N^(-4)*cf
       + 3*N^(-3)*cf
       - 5*N^(-2)*cf
       - 3*N^(-2)*z2*cf
      ;

L   C2q2R =
       - 20*N^(-4)*cf*nf
       + 10*N^(-4)*cf^2
       + 4/3*N^(-3)*cf*nf
       - 55/3*N^(-3)*cf*ca
       + 18*N^(-3)*cf^2
       - 60*N^(-2)*cf*nf
       + 32*N^(-2)*cf*ca
       - 17*N^(-2)*cf^2
       + 16*N^(-2)*z2*cf*nf
       - 24*N^(-2)*z2*cf^2
      ;

L   C2q2e1R =
       + 52*N^(-5)*cf*nf
       - 26*N^(-5)*cf^2
       - 20/3*N^(-4)*cf*nf
       + 143/3*N^(-4)*cf*ca
       - 50*N^(-4)*cf^2
       + 1504/9*N^(-3)*cf*nf
       - 562/9*N^(-3)*cf*ca
       + 47*N^(-3)*cf^2
       - 56*N^(-3)*z2*cf*nf
       + 68*N^(-3)*z2*cf^2
      ;

L   C2q3R =
       + 240*N^(-6)*cf*ca*nf
       - 240*N^(-6)*cf^2*nf
       + 60*N^(-6)*cf^3
       - 368/9*N^(-5)*cf*nf^2
       + 3416/9*N^(-5)*cf*ca*nf
       - 956/9*N^(-5)*cf^2*nf
       - 2002/9*N^(-5)*cf^2*ca
       + 134*N^(-5)*cf^3
       + 656/9*N^(-4)*cf*nf^2
       + 14960/27*N^(-4)*cf*ca*nf
       + 5566/27*N^(-4)*cf*ca^2
       - 5158/9*N^(-4)*cf^2*nf
       + 835/9*N^(-4)*cf^2*ca
       - 30*N^(-4)*cf^3
       - 320/3*N^(-4)*z2*cf*ca*nf
       - 120*N^(-4)*z2*cf*ca^2
       + 1328/3*N^(-4)*z2*cf^2*nf
       + 384*N^(-4)*z2*cf^2*ca
       - 524*N^(-4)*z2*cf^3
      ;

* eq.(A.7)
L   C2g1R =
       + 2*N^(-2)*nf
       - 2*N^(-1)*nf
       + 6*nf
       - 2*z2*nf
      ;

L   C2g1e1R =
       - 2*N^(-3)*nf
       + 2*N^(-2)*nf
       - 6*N^(-1)*nf
       + 3*N^(-1)*z2*nf
      ;

L   C2g1e2R =
       + 2*N^(-4)*nf
       - 2*N^(-3)*nf
       + 6*N^(-2)*nf
       - 3*N^(-2)*z2*nf
      ;

L   C2g2R =
       - 20*N^(-4)*ca*nf
       + 10*N^(-4)*cf*nf
       - 2*N^(-3)*ca*nf
       - 3*N^(-3)*cf*nf
       - 58*N^(-2)*ca*nf
       + 16*N^(-2)*cf*nf
       + 8*N^(-2)*z2*ca*nf
       - 16*N^(-2)*z2*cf*nf
      ;

L   C2g2e1R =
       + 52*N^(-5)*ca*nf
       - 26*N^(-5)*cf*nf
       + 2*N^(-4)*ca*nf
       + 3*N^(-4)*cf*nf
       + 166*N^(-3)*ca*nf
       - 20*N^(-3)*cf*nf
       - 32*N^(-3)*z2*ca*nf
       + 44*N^(-3)*z2*cf*nf
      ;

L   C2g3R =
       + 240*N^(-6)*ca^2*nf
       - 120*N^(-6)*cf*nf^2
       - 120*N^(-6)*cf*ca*nf
       + 60*N^(-6)*cf^2*nf
       - 8/9*N^(-5)*ca*nf^2
       + 1436/9*N^(-5)*ca^2*nf
       + 1636/9*N^(-5)*cf*nf^2
       - 1636/9*N^(-5)*cf*ca*nf
       + 44/3*N^(-5)*cf^2*nf
       + 532/27*N^(-4)*ca*nf^2
       + 27338/27*N^(-4)*ca^2*nf
       - 17782/27*N^(-4)*cf*nf^2
       - 4589/27*N^(-4)*cf*ca*nf
       + 178/3*N^(-4)*cf^2*nf
       - 56*N^(-4)*z2*ca^2*nf
       + 88*N^(-4)*z2*cf*nf^2
       + 656/3*N^(-4)*z2*cf*ca*nf
       - 524/3*N^(-4)*z2*cf^2*nf
      ;

* eq.(A.8)
L   Cpq1R =
       - 4*N^(-2)*cf
       - 4*N^(-1)*cf
       + 5*cf
       + 4*z2*cf
      ;

L   Cpq1e1R =
       + 4*N^(-3)*cf
       + 4*N^(-2)*cf
       + N^(-1)*cf
       - 6*N^(-1)*z2*cf
      ;

L   Cpq1e2R =
       - 4*N^(-4)*cf
       - 4*N^(-3)*cf
       - N^(-2)*cf
       + 6*N^(-2)*z2*cf
      ;

L   Cpq2R =
       + 40*N^(-4)*cf*ca
       - 20*N^(-4)*cf^2
       - 32/3*N^(-3)*cf*nf
       + 344/3*N^(-3)*cf*ca
       - 28*N^(-3)*cf^2
       + 12*N^(-2)*cf*nf
       + 16*N^(-2)*cf*ca
       + 21*N^(-2)*cf^2
       - 16*N^(-2)*z2*cf*ca
       + 32*N^(-2)*z2*cf^2
      ;

L   Cpq2e1R =
       - 104*N^(-5)*cf*ca
       + 52*N^(-5)*cf^2
       + 32*N^(-4)*cf*nf
       - 328*N^(-4)*cf*ca
       + 76*N^(-4)*cf^2
       - 196/9*N^(-3)*cf*nf
       - 1196/9*N^(-3)*cf*ca
       - 25*N^(-3)*cf^2
       + 80*N^(-3)*z2*cf*ca
       - 104*N^(-3)*z2*cf^2
      ;

L   Cpq3R =
       - 480*N^(-6)*cf*ca^2
       + 240*N^(-6)*cf^2*nf
       + 240*N^(-6)*cf^2*ca
       - 120*N^(-6)*cf^3
       + 1072/9*N^(-5)*cf*ca*nf
       - 13960/9*N^(-5)*cf*ca^2
       - 440/9*N^(-5)*cf^2*nf
       + 8960/9*N^(-5)*cf^2*ca
       - 224*N^(-5)*cf^3
       - 32*N^(-4)*cf*nf^2
       + 6592/27*N^(-4)*cf*ca*nf
       - 69928/27*N^(-4)*cf*ca^2
       + 17456/27*N^(-4)*cf^2*nf
       + 2338/27*N^(-4)*cf^2*ca
       + 308/3*N^(-4)*cf^3
       + 208/3*N^(-4)*z2*cf*ca^2
       - 176*N^(-4)*z2*cf^2*nf
       - 1120/3*N^(-4)*z2*cf^2*ca
       + 328*N^(-4)*z2*cf^3
      ;

* eq.(A.9)
L   Cpg1R =
       - 4*N^(-2)*ca
       + 2/3*N^(-1)*nf
       - 23/3*N^(-1)*ca
       - 16/9*nf
       + 118/9*ca
       + 4*z2*ca
      ;

L   Cpg1e1R =
       + 4*N^(-3)*ca
       - 2/3*N^(-2)*nf
       + 23/3*N^(-2)*ca
       + 16/9*N^(-1)*nf
       - 64/9*N^(-1)*ca
       - 6*N^(-1)*z2*ca
      ;

L   Cpg1e2R =
       - 4*N^(-4)*ca
       + 2/3*N^(-3)*nf
       - 23/3*N^(-3)*ca
       - 16/9*N^(-2)*nf
       + 64/9*N^(-2)*ca
       + 6*N^(-2)*z2*ca
      ;

L   Cpg2R =
       + 40*N^(-4)*ca^2
       - 20*N^(-4)*cf*nf
       - 4*N^(-3)*ca*nf
       + 78*N^(-3)*ca^2
       + 14*N^(-3)*cf*nf
       + 8/9*N^(-2)*nf^2
       - 22/9*N^(-2)*ca*nf
       + 833/9*N^(-2)*ca^2
       - 34*N^(-2)*cf*nf
       + 16*N^(-2)*z2*cf*nf
      ;

L   Cpg2e1R =
       - 104*N^(-5)*ca^2
       + 52*N^(-5)*cf*nf
       + 44/3*N^(-4)*ca*nf
       - 698/3*N^(-4)*ca^2
       - 30*N^(-4)*cf*nf
       - 8/3*N^(-3)*nf^2
       + 142/9*N^(-3)*ca*nf
       - 2857/9*N^(-3)*ca^2
       + 94*N^(-3)*cf*nf
       + 32*N^(-3)*z2*ca^2
       - 56*N^(-3)*z2*cf*nf
      ;

L   Cpg3R =
       - 480*N^(-6)*ca^3
       + 480*N^(-6)*cf*ca*nf
       - 120*N^(-6)*cf^2*nf
       + 352/9*N^(-5)*ca^2*nf
       - 10000/9*N^(-5)*ca^3
       - 536/9*N^(-5)*cf*nf^2
       + 3140/9*N^(-5)*cf*ca*nf
       + 44/3*N^(-5)*cf^2*nf
       - 328/27*N^(-4)*ca*nf^2
       + 560/27*N^(-4)*ca^2*nf
       - 59902/27*N^(-4)*ca^3
       + 3508/27*N^(-4)*cf*nf^2
       + 16622/27*N^(-4)*cf*ca*nf
       - 162*N^(-4)*cf^2*nf
       - 48*N^(-4)*z2*ca^2*nf
       - 224*N^(-4)*z2*ca^3
       - 256/3*N^(-4)*z2*cf*ca*nf
       + 616/3*N^(-4)*z2*cf^2*nf
      ;

* eq.(A.10)
L   CLq2R =
       - 16*N^(-2)*cf*nf
       + 8*N^(-2)*cf^2
       + 8/3*N^(-1)*cf*nf
       - 44/3*N^(-1)*cf*ca
       + 12*N^(-1)*cf^2
       + 68/9*cf*nf
       + 538/9*cf*ca
       - 74*cf^2
       + 16*z2*cf*nf
       - 8*z2*cf^2
      ;

L   CLq2e1R =
       + 16*N^(-3)*cf*nf
       - 8*N^(-3)*cf^2
       - 104/3*N^(-2)*cf*nf
       + 44/3*N^(-2)*cf*ca
       - 4*N^(-2)*cf^2
       + 748/9*N^(-1)*cf*nf
       - 670/9*N^(-1)*cf*ca
       + 70*N^(-1)*cf^2
       - 40*N^(-1)*z2*cf*nf
       + 20*N^(-1)*z2*cf^2
      ;

L   CLq3R =
       + 160*N^(-4)*cf*ca*nf
       - 160*N^(-4)*cf^2*nf
       + 40*N^(-4)*cf^3
       - 64/3*N^(-3)*cf*nf^2
       + 496/3*N^(-3)*cf*ca*nf
       + 8*N^(-3)*cf^2*nf
       - 132*N^(-3)*cf^2*ca
       + 64*N^(-3)*cf^3
       + 544/9*N^(-2)*cf*nf^2
       + 16/3*N^(-2)*cf*ca*nf
       + 968/9*N^(-2)*cf*ca^2
       - 944/9*N^(-2)*cf^2*nf
       + 1832/9*N^(-2)*cf^2*ca
       - 168*N^(-2)*cf^3
       - 112*N^(-2)*z2*cf*ca*nf
       - 120*N^(-2)*z2*cf*ca^2
       + 256*N^(-2)*z2*cf^2*nf
       + 384*N^(-2)*z2*cf^2*ca
       - 416*N^(-2)*z2*cf^3
      ;

* eq.(A.11)
L   CLg1R =
       + 4*nf
       - 6*N*nf
       + 7*N^2*nf
      ;

L   CLg1e1R =
       + 8*nf
       - 12*N*nf
       + 4*N*z2*nf
      ;

L   CLg1e2R =
       + 16*nf
       - 2*z2*nf
      ;

L   CLg2R =
       - 16*N^(-2)*ca*nf
       + 8*N^(-2)*cf*nf
       - 8*N^(-1)*cf*nf
       + 16*ca*nf
       - 4*cf*nf
       + 16*z2*ca*nf
       - 8*z2*cf*nf
      ;

L   CLg2e1R =
       + 16*N^(-3)*ca*nf
       - 8*N^(-3)*cf*nf
       - 32*N^(-2)*ca*nf
       + 16*N^(-2)*cf*nf
       + 72*N^(-1)*ca*nf
       - 12*N^(-1)*cf*nf
       - 40*N^(-1)*z2*ca*nf
       + 20*N^(-1)*z2*cf*nf
      ;

L   CLg3R =
       + 160*N^(-4)*ca^2*nf
       - 80*N^(-4)*cf*nf^2
       - 80*N^(-4)*cf*ca*nf
       + 40*N^(-4)*cf^2*nf
       + 16/3*N^(-3)*ca*nf^2
       + 56/3*N^(-3)*ca^2*nf
       + 464/3*N^(-3)*cf*nf^2
       - 152/3*N^(-3)*cf*ca*nf
       - 20*N^(-3)*cf^2*nf
       + 80/9*N^(-2)*ca*nf^2
       + 3640/9*N^(-2)*ca^2*nf
       - 3416/9*N^(-2)*cf*nf^2
       + 308/9*N^(-2)*cf*ca*nf
       - 16*N^(-2)*cf^2*nf
       - 120*N^(-2)*z2*ca^2*nf
       + 64*N^(-2)*z2*cf*nf^2
       + 144*N^(-2)*z2*cf*ca*nf
       - 96*N^(-2)*z2*cf^2*nf
      ;

* -----------------------------------------------------------------------------
