g3nsp := N[gqq3nspFitted + gqq3nspN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nspFitted = 2404.7617993125536 + 47399.00434062458/(1 + n)^3 - 
     15176.296853013831/(1 + n)^2 - 11103.411980157494/(2 + n) - 
     43731.12143482942*Lm11m1[n] - 2518.9090401926924*Lm12m1[n] - 
     973.3270027901576*Lm13m1[n] - (73498.98594171858*S[1, n])/n^2 + 
     nf^2*(-1.290836887384268 + 59.46630017646719/(1 + n)^3 - 
       152.70402416764668/(1 + n)^2 - 94.57207315818547/(2 + n) + 
       1.5006487633206929*Lm11m1[n] + 113.48340560825889*Lm12m1[n] + 
       13.865450025251006*Lm13m1[n] - (517.9354004395117*S[1, n])/n^2) + 
     nf^3*(481.21845075689777 - 1.0804850259395624/(1 + n)^3 + 
       3.883615725797128/(1 + n)^2 - 1.619324141241275/(2 + n) - 
       0.5794585257028956*Lm11m1[n] + 0.36678601773373637*Lm12m1[n] + 
       0.052348537749681484*Lm13m1[n] - (1.8801708330418763*S[1, n])/n^2) + 
     nf*(-268.3515573005987 + 537.8609133198307/(1 + n)^3 - 
       718.3874592628895/(1 + n)^2 + 2487.96294221855/(2 + n) - 
       849.8232086542307*Lm11m1[n] - 3106.3285877376907*Lm12m1[n] - 
       399.22204467960154*Lm13m1[n] + (12894.65275887218*S[1, n])/n^2)
 
Lm11m1[n_Real] := 1/(1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/
      (1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/(n*(1 + n)^2)
 
Lm12m1[n_Real] := -(2/(1 + n)^3) - (2*(EulerGamma + PolyGamma[0, 1 + n]))/
      (1 + n)^2 + (EulerGamma + PolyGamma[0, 1 + n])^2/n - 
     (EulerGamma + PolyGamma[0, 1 + n])^2/(1 + n) + 
     (Pi^2/6 - PolyGamma[1, 1 + n])/n - (Pi^2/6 - PolyGamma[1, 1 + n])/(1 + n)
 
Lm13m1[n_Real] := (-(1/(2*n*(1 + n))))*(-6*EulerGamma^2 + 2*EulerGamma^3 - 
      Pi^2 + EulerGamma*Pi^2 + 6*(-1 + EulerGamma)*PolyGamma[0, 2 + n]^2 + 
      2*PolyGamma[0, 2 + n]^3 + PolyGamma[0, 2 + n]*(-12*EulerGamma + 
        6*EulerGamma^2 + Pi^2 - 6*PolyGamma[1, 2 + n]) - 
      6*(-1 + EulerGamma)*PolyGamma[1, 2 + n] + 2*PolyGamma[2, 2 + n] + 
      4*Zeta[3])
 
gqq3nspN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
     (128*cf^4)/n^5 + (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^6 - 
     (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^5 - 
     (24*cf^2*((11*ca)/3 - (2*nf)/3)^2)/n^5 + (480*ca^2*cf^2*z2)/n^5 - 
     (1536*ca*cf^3*z2)/n^5 + (1600*cf^4*z2)/n^5 - 
     ((-2662*ca^3*cf)/27 - (13060*ca^2*cf^2)/9 - 20*ca*cf^3 + 212*cf^4 + 
       (484*ca^2*cf*nf)/9 + (4184*ca*cf^2*nf)/9 + 32*cf^3*nf - 
       (88*ca*cf*nf^2)/9 - (304*cf^2*nf^2)/9 + (16*cf*nf^3)/27 + 
       192*cf*nc^3*z2 - 48*cf*nc^2*nf*z2 - 288*ca*cf^3*z3 + 640*cf^4*z3)/
      n^4 - ((50006*ca^3*cf)/81 + (229480*ca^2*cf^2)/81 + 196*ca*cf^3 - 
       224*cf^4 - (2780*ca^2*cf*nf)/9 - (65936*ca*cf^2*nf)/81 - 
       (340*cf^3*nf)/3 + (1288*ca*cf*nf^2)/27 + (4288*cf^2*nf^2)/81 - 
       (176*cf*nf^3)/81 - (7682*cf*nc^3*z2)/9 + (1552*cf*nc^2*nf*z2)/9 - 
       (80*cf*nc*nf^2*z2)/9 + 64*ca^2*cf^2*z3 + (832*ca*cf^3*z3)/3 - 
       512*cf^4*z3 - 128*ca*cf^2*nf*z3 + (512*cf^3*nf*z3)/3 + 236*cf*nc^3*z4)/
      n^3 - ((-146482*ca^3*cf)/81 - (254225*ca^2*cf^2)/81 + 
       (2761*ca*cf^3)/3 + 130*cf^4 + (64481*ca^2*cf*nf)/81 + 
       (90538*ca*cf^2*nf)/81 - (500*cf^3*nf)/3 - (7561*ca*cf*nf^2)/81 - 
       (7736*cf^2*nf^2)/81 + (64*cf*nf^3)/27 + (12221*cf*nc^3*z2)/9 - 
       (4006*cf*nc^2*nf*z2)/9 + (272*cf*nc*nf^2*z2)/9 + 264*ca^3*cf*z3 - 
       (8984*ca^2*cf^2*z3)/3 + (12448*ca*cf^3*z3)/3 + 944*cf^4*z3 + 
       (32*ca^2*cf*nf*z3)/3 + (1328*ca*cf^2*nf*z3)/3 - (2080*cf^3*nf*z3)/3 - 
       (32*ca*cf*nf^2*z3)/3 - 48*cf*nc^3*z2*z3 - (1312*cf*nc^3*z4)/3 + 
       (328*cf*nc^2*nf*z4)/3 + 240*ca^2*cf^2*z5 + 960*ca*cf^3*z5 - 
       1920*cf^4*z5)/n^2
 
Attributes[z2] = {Constant}
 
N[z2, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[2], HarmonicSums`Private`b]
 
Attributes[z3] = {Constant}
 
N[z3, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[3], HarmonicSums`Private`b]
 
Attributes[z5] = {Constant}
 
N[z5, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[5], HarmonicSums`Private`b]
 
gqq3nsNinf = (4*nf^3*(131/81 - (32*z2)/81 - 304*z3 + (32*z4)/27))/3 + 
     4*nf^2*(127/18 - (5036*z2)/81 + (932*z3)/27 - (160*z2*z3)/9 + 
       (1292*z4)/27 - (32*z5)/3) - 
     (64*(-7 + (2*nf)/3)*(18.357785497334497 - 2.8623036006584215*nf - 
         0.012345679012345678*nf^2) - 128*(80.86856651940118 - 
         20.20279737924136*nf + 0.763973538214516*nf^2 + 
         0.012782595512258363*nf^3) + 16*(4.154576310747992 - 
         0.37037037037037035*nf)*(-102 + (38*nf)/3 - 
         (16*nf*(1/12 + (2*z2)/3))/3 + 16*(17/24 + (11*z2)/3 - 3*z3) + 
         (64*(3/8 - 3*z2 + 6*z3))/9) + 
       (16*(-94237/54 + (5737*nf)/18 - (461*nf^2)/54 + 1312*z2 - 
          (5024*nf*z2)/27 + (320*nf^2*z2)/81 - (10744*z2^2)/45 + 
          (4144*nf*z2^2)/135 - (11008*z3)/27 + (224*nf*z3)/27 - 
          (64*nf^2*z3)/27 + (256*z2*z3)/27 + (4960*z5)/9))/3)/n - 
     12*nf*(353/3 - (85175*z2)/162 - (137*z3)/9 - (584*z2*z3)/9 - 
       (16*z3^2)/3 + (16186*z4)/27 - (248*z5)/3 - 144*z6) - 
     36*(-1379569/5184 + (24211*z2)/27 - (9803*z3)/162 + (838*z2*z3)/9 + 
       (16*z3^2)/3 - (9382*z4)/9 + 32*z3*z4 + 1002*z5 - 80*z2*z5 + 135*z6 - 
       560*z7) + 256*(80.86856651940118 - 20.20279737924136*nf + 
       0.763973538214516*nf^2 + 0.012782595512258363*nf^3)*S[1, n] + 
     ((256*(4.154576310747992 - 0.37037037037037035*nf)^2 + 
        (2048*(18.357785497334497 - 2.8623036006584215*nf - 
           0.012345679012345678*nf^2))/3)*S[1, n])/n
 
Attributes[z7] = {Constant}
 
N[z7, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[7], HarmonicSums`Private`b]
 
QCDConstantsRules = {ca -> 3, nc -> 3, cf -> 4/3, tr -> 1/2, d4RA/nr -> 5/2, 
     d4RR/nr -> 5/36, d4AA/na -> (nc^2*(36 + nc^2))/24, 
     d4RA/na -> (nc*(6 + nc^2))/48, d4RR/na -> (18 - 6*nc^2 + nc^4)/
       (96*nc^2), caf -> ca - cf}
 
ZetaRules = {z2 -> Pi^2/6, z3 -> Zeta[3], z4 -> Pi^4/90, z5 -> Zeta[5], 
     z6 -> Pi^6/945, z7 -> Zeta[7]}
g3nsp := N[gqq3nspFitted + gqq3nspN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nspFitted = -46785.12166323133 + 47399.001814374584/(1 + n)^3 - 
     15176.29406955748/(1 + n)^2 - 11103.411800806844/(2 + n) - 
     43731.11790490434*Lm11m1[n] - 2518.907643518572*Lm12m1[n] - 
     973.3266732717074*Lm13m1[n] - (73498.9857181551*S[1, n])/n^2 + 
     nf^2*(-387.7175279940118 + 59.46630262121647/(1 + n)^3 - 
       152.70402673087884/(1 + n)^2 - 94.57207329417675/(2 + n) + 
       1.500645520763423*Lm11m1[n] + 113.48340435254481*Lm12m1[n] + 
       13.865449748118452*Lm13m1[n] - (517.935400666252*S[1, n])/n^2) + 
     nf^3*(-6.030013461509355 - 1.080478775351189/(1 + n)^3 + 
       3.8836088198113274/(1 + n)^2 - 1.6193245860301646/(2 + n) - 
       0.5794672800992433*Lm11m1[n] + 0.3667825504805495*Lm12m1[n] + 
       0.052347719718235555*Lm13m1[n] - (1.8801713842271066*S[1, n])/n^2) + 
     nf*(11100.329839459302 + 537.860424471822/(1 + n)^3 - 
       718.3869297551672/(1 + n)^2 + 2487.962974739384/(2 + n) - 
       849.8225371856591*Lm11m1[n] - 3106.3283238846625*Lm12m1[n] - 
       399.2219834871794*Lm13m1[n] + (12894.652802909668*S[1, n])/n^2)
 
Lm11m1[n_Real] := 1/(1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/
      (1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/(n*(1 + n)^2)
 
Lm12m1[n_Real] := -(2/(1 + n)^3) - (2*(EulerGamma + PolyGamma[0, 1 + n]))/
      (1 + n)^2 + (EulerGamma + PolyGamma[0, 1 + n])^2/n - 
     (EulerGamma + PolyGamma[0, 1 + n])^2/(1 + n) + 
     (Pi^2/6 - PolyGamma[1, 1 + n])/n - (Pi^2/6 - PolyGamma[1, 1 + n])/(1 + n)
 
Lm13m1[n_Real] := -((1/(2*n*(1 + n)))*(-6*EulerGamma^2 + 2*EulerGamma^3 - 
       Pi^2 + EulerGamma*Pi^2 + 6*(-1 + EulerGamma)*PolyGamma[0, 2 + n]^2 + 
       2*PolyGamma[0, 2 + n]^3 + PolyGamma[0, 2 + n]*(-12*EulerGamma + 
         6*EulerGamma^2 + Pi^2 - 6*PolyGamma[1, 2 + n]) - 
       6*(-1 + EulerGamma)*PolyGamma[1, 2 + n] + 2*PolyGamma[2, 2 + n] + 
       4*Zeta[3]))
 
gqq3nspN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
     (128*cf^4)/n^5 + (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^6 - 
     (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^5 - 
     (24*cf^2*((11*ca)/3 - (2*nf)/3)^2)/n^5 + (480*ca^2*cf^2*z2)/n^5 - 
     (1536*ca*cf^3*z2)/n^5 + (1600*cf^4*z2)/n^5 - 
     ((-2662*ca^3*cf)/27 - (13060*ca^2*cf^2)/9 - 20*ca*cf^3 + 212*cf^4 + 
       (484*ca^2*cf*nf)/9 + (4184*ca*cf^2*nf)/9 + 32*cf^3*nf - 
       (88*ca*cf*nf^2)/9 - (304*cf^2*nf^2)/9 + (16*cf*nf^3)/27 + 
       192*cf*nc^3*z2 - 48*cf*nc^2*nf*z2 - 288*ca*cf^3*z3 + 640*cf^4*z3)/
      n^4 - ((50006*ca^3*cf)/81 + (229480*ca^2*cf^2)/81 + 196*ca*cf^3 - 
       224*cf^4 - (2780*ca^2*cf*nf)/9 - (65936*ca*cf^2*nf)/81 - 
       (340*cf^3*nf)/3 + (1288*ca*cf*nf^2)/27 + (4288*cf^2*nf^2)/81 - 
       (176*cf*nf^3)/81 - (7682*cf*nc^3*z2)/9 + (1552*cf*nc^2*nf*z2)/9 - 
       (80*cf*nc*nf^2*z2)/9 + 64*ca^2*cf^2*z3 + (832*ca*cf^3*z3)/3 - 
       512*cf^4*z3 - 128*ca*cf^2*nf*z3 + (512*cf^3*nf*z3)/3 + 236*cf*nc^3*z4)/
      n^3 - ((-146482*ca^3*cf)/81 - (254225*ca^2*cf^2)/81 + 
       (2761*ca*cf^3)/3 + 130*cf^4 + (64481*ca^2*cf*nf)/81 + 
       (90538*ca*cf^2*nf)/81 - (500*cf^3*nf)/3 - (7561*ca*cf*nf^2)/81 - 
       (7736*cf^2*nf^2)/81 + (64*cf*nf^3)/27 + (12221*cf*nc^3*z2)/9 - 
       (4006*cf*nc^2*nf*z2)/9 + (272*cf*nc*nf^2*z2)/9 + 264*ca^3*cf*z3 - 
       (8984*ca^2*cf^2*z3)/3 + (12448*ca*cf^3*z3)/3 + 944*cf^4*z3 + 
       (32*ca^2*cf*nf*z3)/3 + (1328*ca*cf^2*nf*z3)/3 - (2080*cf^3*nf*z3)/3 - 
       (32*ca*cf*nf^2*z3)/3 - 48*cf*nc^3*z2*z3 - (1312*cf*nc^3*z4)/3 + 
       (328*cf*nc^2*nf*z4)/3 + 240*ca^2*cf^2*z5 + 960*ca*cf^3*z5 - 
       1920*cf^4*z5)/n^2
 
Attributes[z2] = {Constant}
 
N[z2, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[2], HarmonicSums`Private`b]
 
Attributes[z3] = {Constant}
 
N[z3, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[3], HarmonicSums`Private`b]
 
Attributes[z5] = {Constant}
 
N[z5, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[5], HarmonicSums`Private`b]
 
gqq3nsNinf = -((64*(-7 + (2*nf)/3)*(18.357785497334497 - 2.8623036006584215*
           nf - 0.012345679012345678*nf^2) - 128*(80.86856651940118 - 
          20.20279737924136*nf + 0.763973538214516*nf^2 + 
          0.012782595512258363*nf^3) + 16*(4.154576310747992 - 
          0.37037037037037035*nf)*(-102 + (38*nf)/3 - 
          (16*nf*(1/12 + (2*z2)/3))/3 + 16*(17/24 + (11*z2)/3 - 3*z3) + 
          (64*(3/8 - 3*z2 + 6*z3))/9) + 
        (16*(-94237/54 + (5737*nf)/18 - (461*nf^2)/54 + 1312*z2 - 
           (5024*nf*z2)/27 + (320*nf^2*z2)/81 - (10744*z2^2)/45 + 
           (4144*nf*z2^2)/135 - (11008*z3)/27 + (224*nf*z3)/27 - 
           (64*nf^2*z3)/27 + (256*z2*z3)/27 + (4960*z5)/9))/3)/n) + 
     256*(80.86856651940118 - 20.20279737924136*nf + 0.763973538214516*nf^2 + 
       0.012782595512258363*nf^3)*S[1, n] + 
     ((256*(4.154576310747992 - 0.37037037037037035*nf)^2 + 
        (2048*(18.357785497334497 - 2.8623036006584215*nf - 
           0.012345679012345678*nf^2))/3)*S[1, n])/n - 
     128*(-182.76411280200784 - 13.129625578703704*nf + (3241*nf^2)/15552 + 
       (131*nf^3)/7776 + (85175*nf*Pi^2)/10368 - (7243*nf^2*Pi^2)/23328 - 
       (nf^3*Pi^2)/1458 - (8093*nf*Pi^4)/12960 + (661*nf^2*Pi^4)/43740 + 
       (nf^3*Pi^4)/7290 + (nf*Pi^6)/70 + (137*nf*Zeta[3])/96 + 
       (263*nf^2*Zeta[3])/243 - (19*nf^3*Zeta[3])/486 + 
       (73*nf*Pi^2*Zeta[3])/72 - (95*nf^2*Pi^2*Zeta[3])/972 + 
       (nf*Zeta[3]^2)/2 + (31*nf*Zeta[5])/4 - (85*nf^2*Zeta[5])/324)
 
QCDConstantsRules = {ca -> 3, nc -> 3, cf -> 4/3, tr -> 1/2, d4RA/nr -> 5/2, 
     d4RR/nr -> 5/36, d4AA/na -> (nc^2*(36 + nc^2))/24, 
     d4RA/na -> (nc*(6 + nc^2))/48, d4RR/na -> (18 - 6*nc^2 + nc^4)/
       (96*nc^2), caf -> ca - cf}
 
ZetaRules = {z2 -> Pi^2/6, z3 -> Zeta[3], z4 -> Pi^4/90, z5 -> Zeta[5], 
     z6 -> Pi^6/945, z7 -> Zeta[7]}
 
Attributes[z7] = {Constant}
 
N[z7, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[7], HarmonicSums`Private`b]
g3nsp := N[gqq3nspFitted + gqq3nspN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nspFitted = 2.4912137210289953 + 47399.00038875833/(1 + n)^3 - 
     15176.292500014064/(1 + n)^2 - 11103.411701639641/(2 + n) - 
     43731.115916261864*Lm11m1[n] - 2518.9068570981576*Lm12m1[n] - 
     973.3264889703381*Lm13m1[n] - (73498.98559205797*S[1, n])/n^2 + 
     nf^2*(-0.007439648724645256 + 59.46630130480791/(1 + n)^3 - 
       152.70402546462117/(1 + n)^2 - 94.57207324708389/(2 + n) + 
       1.5006471227080511*Lm11m1[n] + 113.48340494886403*Lm12m1[n] + 
       13.865449866007426*Lm13m1[n] - (517.9354005343661*S[1, n])/n^2) + 
     nf^3*(-0.000049403442695856184 - 1.0804787362748196/(1 + n)^3 + 
       3.883608774695053/(1 + n)^2 - 1.6193245892770178/(2 + n) - 
       0.5794673373035815*Lm11m1[n] + 0.36678252743479567*Lm12m1[n] + 
       0.05234771405499409*Lm13m1[n] - (1.8801713875076356*S[1, n])/n^2) + 
     nf*(0.24051668252106234 + 537.8605823453019/(1 + n)^3 - 
       718.387090748267/(1 + n)^2 + 2487.962966512456/(2 + n) - 
       849.822741421331*Lm11m1[n] - 3106.3284021112195*Lm12m1[n] - 
       399.2220005163702*Lm13m1[n] + (12894.652787822934*S[1, n])/n^2)
 
Lm11m1[n_Real] := 1/(1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/
      (1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/(n*(1 + n)^2)
 
Lm12m1[n_Real] := -(2/(1 + n)^3) - (2*(EulerGamma + PolyGamma[0, 1 + n]))/
      (1 + n)^2 + (EulerGamma + PolyGamma[0, 1 + n])^2/n - 
     (EulerGamma + PolyGamma[0, 1 + n])^2/(1 + n) + 
     (Pi^2/6 - PolyGamma[1, 1 + n])/n - (Pi^2/6 - PolyGamma[1, 1 + n])/(1 + n)
 
Lm13m1[n_Real] := -((1/(2*n*(1 + n)))*(-6*EulerGamma^2 + 2*EulerGamma^3 - 
       Pi^2 + EulerGamma*Pi^2 + 6*(-1 + EulerGamma)*PolyGamma[0, 2 + n]^2 + 
       2*PolyGamma[0, 2 + n]^3 + PolyGamma[0, 2 + n]*(-12*EulerGamma + 
         6*EulerGamma^2 + Pi^2 - 6*PolyGamma[1, 2 + n]) - 
       6*(-1 + EulerGamma)*PolyGamma[1, 2 + n] + 2*PolyGamma[2, 2 + n] + 
       4*Zeta[3]))
 
gqq3nspN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
     (128*cf^4)/n^5 + (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^6 - 
     (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^5 - 
     (24*cf^2*((11*ca)/3 - (2*nf)/3)^2)/n^5 + (480*ca^2*cf^2*z2)/n^5 - 
     (1536*ca*cf^3*z2)/n^5 + (1600*cf^4*z2)/n^5 - 
     ((-2662*ca^3*cf)/27 - (13060*ca^2*cf^2)/9 - 20*ca*cf^3 + 212*cf^4 + 
       (484*ca^2*cf*nf)/9 + (4184*ca*cf^2*nf)/9 + 32*cf^3*nf - 
       (88*ca*cf*nf^2)/9 - (304*cf^2*nf^2)/9 + (16*cf*nf^3)/27 + 
       192*cf*nc^3*z2 - 48*cf*nc^2*nf*z2 - 288*ca*cf^3*z3 + 640*cf^4*z3)/
      n^4 - ((50006*ca^3*cf)/81 + (229480*ca^2*cf^2)/81 + 196*ca*cf^3 - 
       224*cf^4 - (2780*ca^2*cf*nf)/9 - (65936*ca*cf^2*nf)/81 - 
       (340*cf^3*nf)/3 + (1288*ca*cf*nf^2)/27 + (4288*cf^2*nf^2)/81 - 
       (176*cf*nf^3)/81 - (7682*cf*nc^3*z2)/9 + (1552*cf*nc^2*nf*z2)/9 - 
       (80*cf*nc*nf^2*z2)/9 + 64*ca^2*cf^2*z3 + (832*ca*cf^3*z3)/3 - 
       512*cf^4*z3 - 128*ca*cf^2*nf*z3 + (512*cf^3*nf*z3)/3 + 236*cf*nc^3*z4)/
      n^3 - ((-146482*ca^3*cf)/81 - (254225*ca^2*cf^2)/81 + 
       (2761*ca*cf^3)/3 + 130*cf^4 + (64481*ca^2*cf*nf)/81 + 
       (90538*ca*cf^2*nf)/81 - (500*cf^3*nf)/3 - (7561*ca*cf*nf^2)/81 - 
       (7736*cf^2*nf^2)/81 + (64*cf*nf^3)/27 + (12221*cf*nc^3*z2)/9 - 
       (4006*cf*nc^2*nf*z2)/9 + (272*cf*nc*nf^2*z2)/9 + 264*ca^3*cf*z3 - 
       (8984*ca^2*cf^2*z3)/3 + (12448*ca*cf^3*z3)/3 + 944*cf^4*z3 + 
       (32*ca^2*cf*nf*z3)/3 + (1328*ca*cf^2*nf*z3)/3 - (2080*cf^3*nf*z3)/3 - 
       (32*ca*cf*nf^2*z3)/3 - 48*cf*nc^3*z2*z3 - (1312*cf*nc^3*z4)/3 + 
       (328*cf*nc^2*nf*z4)/3 + 240*ca^2*cf^2*z5 + 960*ca*cf^3*z5 - 
       1920*cf^4*z5)/n^2
 
Attributes[z2] = {Constant}
 
N[z2, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[2], HarmonicSums`Private`b]
 
Attributes[z3] = {Constant}
 
N[z3, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[3], HarmonicSums`Private`b]
 
Attributes[z5] = {Constant}
 
N[z5, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[5], HarmonicSums`Private`b]
 
gqq3nsNinf = -((64*(-7 + (2*nf)/3)*(18.357785497334497 - 2.8623036006584215*
           nf - 0.012345679012345678*nf^2) - 128*(80.86856651940118 - 
          20.20279737924136*nf + 0.763973538214516*nf^2 + 
          0.012782595512258363*nf^3) + 16*(4.154576310747992 - 
          0.37037037037037035*nf)*(-102 + (38*nf)/3 - 
          (16*nf*(1/12 + (2*z2)/3))/3 + 16*(17/24 + (11*z2)/3 - 3*z3) + 
          (64*(3/8 - 3*z2 + 6*z3))/9) + 
        (16*(-94237/54 + (5737*nf)/18 - (461*nf^2)/54 + 1312*z2 - 
           (5024*nf*z2)/27 + (320*nf^2*z2)/81 - (10744*z2^2)/45 + 
           (4144*nf*z2^2)/135 - (11008*z3)/27 + (224*nf*z3)/27 - 
           (64*nf^2*z3)/27 + (256*z2*z3)/27 + (4960*z5)/9))/3)/n) + 
     256*(80.86856651940118 - 20.20279737924136*nf + 0.763973538214516*nf^2 + 
       0.012782595512258363*nf^3)*S[1, n] + 
     ((256*(4.154576310747992 - 0.37037037037037035*nf)^2 + 
        (2048*(18.357785497334497 - 2.8623036006584215*nf - 
           0.012345679012345678*nf^2))/3)*S[1, n])/n + 
     128*(-182.76411280200784 - 13.129625578703704*nf + (3241*nf^2)/15552 + 
       (131*nf^3)/7776 + (85175*nf*Pi^2)/10368 - (7243*nf^2*Pi^2)/23328 - 
       (nf^3*Pi^2)/1458 - (8093*nf*Pi^4)/12960 + (661*nf^2*Pi^4)/43740 + 
       (nf^3*Pi^4)/7290 + (nf*Pi^6)/70 + (137*nf*Zeta[3])/96 + 
       (263*nf^2*Zeta[3])/243 - (19*nf^3*Zeta[3])/486 + 
       (73*nf*Pi^2*Zeta[3])/72 - (95*nf^2*Pi^2*Zeta[3])/972 + 
       (nf*Zeta[3]^2)/2 + (31*nf*Zeta[5])/4 - (85*nf^2*Zeta[5])/324)
 
QCDConstantsRules = {ca -> 3, nc -> 3, cf -> 4/3, tr -> 1/2, d4RA/nr -> 5/2, 
     d4RR/nr -> 5/36, d4AA/na -> (nc^2*(36 + nc^2))/24, 
     d4RA/na -> (nc*(6 + nc^2))/48, d4RR/na -> (18 - 6*nc^2 + nc^4)/
       (96*nc^2), caf -> ca - cf}
 
ZetaRules = {z2 -> Pi^2/6, z3 -> Zeta[3], z4 -> Pi^4/90, z5 -> Zeta[5], 
     z6 -> Pi^6/945, z7 -> Zeta[7]}
 
Attributes[z7] = {Constant}
 
N[z7, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[7], HarmonicSums`Private`b]
g3nsp := N[gqq3nspFitted + gqq3nspN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nspFitted = 2.4912137210289953 + 47399.00038875833/(1 + n)^3 - 
     15176.292500014064/(1 + n)^2 - 11103.411701639641/(2 + n) - 
     43731.115916261864*Lm11m1[n] - 2518.9068570981576*Lm12m1[n] - 
     973.3264889703381*Lm13m1[n] - (73498.98559205797*S[1, n])/n^2 + 
     nf^2*(-0.007439648724645256 + 59.46630130480791/(1 + n)^3 - 
       152.70402546462117/(1 + n)^2 - 94.57207324708389/(2 + n) + 
       1.5006471227080511*Lm11m1[n] + 113.48340494886403*Lm12m1[n] + 
       13.865449866007426*Lm13m1[n] - (517.9354005343661*S[1, n])/n^2) + 
     nf^3*(-0.000049403442695856184 - 1.0804787362748196/(1 + n)^3 + 
       3.883608774695053/(1 + n)^2 - 1.6193245892770178/(2 + n) - 
       0.5794673373035815*Lm11m1[n] + 0.36678252743479567*Lm12m1[n] + 
       0.05234771405499409*Lm13m1[n] - (1.8801713875076356*S[1, n])/n^2) + 
     nf*(0.24051668252106234 + 537.8605823453019/(1 + n)^3 - 
       718.387090748267/(1 + n)^2 + 2487.962966512456/(2 + n) - 
       849.822741421331*Lm11m1[n] - 3106.3284021112195*Lm12m1[n] - 
       399.2220005163702*Lm13m1[n] + (12894.652787822934*S[1, n])/n^2)
 
Lm11m1[n_Real] := 1/(1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/
      (1 + n)^2 - (EulerGamma + PolyGamma[0, 1 + n])/(n*(1 + n)^2)
 
Lm12m1[n_Real] := -(2/(1 + n)^3) - (2*(EulerGamma + PolyGamma[0, 1 + n]))/
      (1 + n)^2 + (EulerGamma + PolyGamma[0, 1 + n])^2/n - 
     (EulerGamma + PolyGamma[0, 1 + n])^2/(1 + n) + 
     (Pi^2/6 - PolyGamma[1, 1 + n])/n - (Pi^2/6 - PolyGamma[1, 1 + n])/(1 + n)
 
Lm13m1[n_Real] := -((1/(2*n*(1 + n)))*(-6*EulerGamma^2 + 2*EulerGamma^3 - 
       Pi^2 + EulerGamma*Pi^2 + 6*(-1 + EulerGamma)*PolyGamma[0, 2 + n]^2 + 
       2*PolyGamma[0, 2 + n]^3 + PolyGamma[0, 2 + n]*(-12*EulerGamma + 
         6*EulerGamma^2 + Pi^2 - 6*PolyGamma[1, 2 + n]) - 
       6*(-1 + EulerGamma)*PolyGamma[1, 2 + n] + 2*PolyGamma[2, 2 + n] + 
       4*Zeta[3]))
 
gqq3nspN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
     (128*cf^4)/n^5 + (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^6 - 
     (80*cf^3*((11*ca)/3 - (2*nf)/3))/n^5 - 
     (24*cf^2*((11*ca)/3 - (2*nf)/3)^2)/n^5 + (480*ca^2*cf^2*z2)/n^5 - 
     (1536*ca*cf^3*z2)/n^5 + (1600*cf^4*z2)/n^5 - 
     ((-2662*ca^3*cf)/27 - (13060*ca^2*cf^2)/9 - 20*ca*cf^3 + 212*cf^4 + 
       (484*ca^2*cf*nf)/9 + (4184*ca*cf^2*nf)/9 + 32*cf^3*nf - 
       (88*ca*cf*nf^2)/9 - (304*cf^2*nf^2)/9 + (16*cf*nf^3)/27 + 
       192*cf*nc^3*z2 - 48*cf*nc^2*nf*z2 - 288*ca*cf^3*z3 + 640*cf^4*z3)/
      n^4 - ((50006*ca^3*cf)/81 + (229480*ca^2*cf^2)/81 + 196*ca*cf^3 - 
       224*cf^4 - (2780*ca^2*cf*nf)/9 - (65936*ca*cf^2*nf)/81 - 
       (340*cf^3*nf)/3 + (1288*ca*cf*nf^2)/27 + (4288*cf^2*nf^2)/81 - 
       (176*cf*nf^3)/81 - (7682*cf*nc^3*z2)/9 + (1552*cf*nc^2*nf*z2)/9 - 
       (80*cf*nc*nf^2*z2)/9 + 64*ca^2*cf^2*z3 + (832*ca*cf^3*z3)/3 - 
       512*cf^4*z3 - 128*ca*cf^2*nf*z3 + (512*cf^3*nf*z3)/3 + 236*cf*nc^3*z4)/
      n^3 - ((-146482*ca^3*cf)/81 - (254225*ca^2*cf^2)/81 + 
       (2761*ca*cf^3)/3 + 130*cf^4 + (64481*ca^2*cf*nf)/81 + 
       (90538*ca*cf^2*nf)/81 - (500*cf^3*nf)/3 - (7561*ca*cf*nf^2)/81 - 
       (7736*cf^2*nf^2)/81 + (64*cf*nf^3)/27 + (12221*cf*nc^3*z2)/9 - 
       (4006*cf*nc^2*nf*z2)/9 + (272*cf*nc*nf^2*z2)/9 + 264*ca^3*cf*z3 - 
       (8984*ca^2*cf^2*z3)/3 + (12448*ca*cf^3*z3)/3 + 944*cf^4*z3 + 
       (32*ca^2*cf*nf*z3)/3 + (1328*ca*cf^2*nf*z3)/3 - (2080*cf^3*nf*z3)/3 - 
       (32*ca*cf*nf^2*z3)/3 - 48*cf*nc^3*z2*z3 - (1312*cf*nc^3*z4)/3 + 
       (328*cf*nc^2*nf*z4)/3 + 240*ca^2*cf^2*z5 + 960*ca*cf^3*z5 - 
       1920*cf^4*z5)/n^2
 
Attributes[z2] = {Constant}
 
N[z2, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[2], HarmonicSums`Private`b]
 
Attributes[z3] = {Constant}
 
N[z3, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[3], HarmonicSums`Private`b]
 
Attributes[z5] = {Constant}
 
N[z5, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[5], HarmonicSums`Private`b]
 
gqq3nsNinf = -((64*(-7 + (2*nf)/3)*(18.357785497334497 - 2.8623036006584215*
           nf - 0.012345679012345678*nf^2) - 128*(80.86856651940118 - 
          20.20279737924136*nf + 0.763973538214516*nf^2 + 
          0.012782595512258363*nf^3) + 16*(4.154576310747992 - 
          0.37037037037037035*nf)*(-102 + (38*nf)/3 - 
          (16*nf*(1/12 + (2*z2)/3))/3 + 16*(17/24 + (11*z2)/3 - 3*z3) + 
          (64*(3/8 - 3*z2 + 6*z3))/9) + 
        (16*(-94237/54 + (5737*nf)/18 - (461*nf^2)/54 + 1312*z2 - 
           (5024*nf*z2)/27 + (320*nf^2*z2)/81 - (10744*z2^2)/45 + 
           (4144*nf*z2^2)/135 - (11008*z3)/27 + (224*nf*z3)/27 - 
           (64*nf^2*z3)/27 + (256*z2*z3)/27 + (4960*z5)/9))/3)/n) + 
     256*(80.86856651940118 - 20.20279737924136*nf + 0.763973538214516*nf^2 + 
       0.012782595512258363*nf^3)*S[1, n] + 
     ((256*(4.154576310747992 - 0.37037037037037035*nf)^2 + 
        (2048*(18.357785497334497 - 2.8623036006584215*nf - 
           0.012345679012345678*nf^2))/3)*S[1, n])/n + 
     128*(-182.76411280200784 - 13.129625578703704*nf + (3241*nf^2)/15552 + 
       (131*nf^3)/7776 + (85175*nf*Pi^2)/10368 - (7243*nf^2*Pi^2)/23328 - 
       (nf^3*Pi^2)/1458 - (8093*nf*Pi^4)/12960 + (661*nf^2*Pi^4)/43740 + 
       (nf^3*Pi^4)/7290 + (nf*Pi^6)/70 + (137*nf*Zeta[3])/96 + 
       (263*nf^2*Zeta[3])/243 - (19*nf^3*Zeta[3])/486 + 
       (73*nf*Pi^2*Zeta[3])/72 - (95*nf^2*Pi^2*Zeta[3])/972 + 
       (nf*Zeta[3]^2)/2 + (31*nf*Zeta[5])/4 - (85*nf^2*Zeta[5])/324)
 
QCDConstantsRules = {ca -> 3, nc -> 3, cf -> 4/3, tr -> 1/2, d4RA/nr -> 5/2, 
     d4RR/nr -> 5/36, d4AA/na -> (nc^2*(36 + nc^2))/24, 
     d4RA/na -> (nc*(6 + nc^2))/48, d4RR/na -> (18 - 6*nc^2 + nc^4)/
       (96*nc^2), caf -> ca - cf}
 
ZetaRules = {z2 -> Pi^2/6, z3 -> Zeta[3], z4 -> Pi^4/90, z5 -> Zeta[5], 
     z6 -> Pi^6/945, z7 -> Zeta[7]}
 
Attributes[z7] = {Constant}
 
N[z7, HarmonicSums`Private`b_:{MachinePrecision, MachinePrecision}] := 
    N[Zeta[7], HarmonicSums`Private`b]
