g3nsm := N[gqq3nsmFitted + gqq3nsmN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nsmFitted = 2424.06511308964 + 194339.87834020052/(1 + n)^3 - 
     88491.39062175922/(1 + n)^2 - 16673.930496518376/(3 + n) - 
     178815.0878250944*Lm11m1[n] - 49111.66189344577*Lm12m1[n] - 
     11804.70644702107*Lm13m1[n] - (103246.60425090564*S[1, n])/n^2 + 
     nf^2*(-1.318464648031682 - 192.0247446101425/(1 + n)^3 + 
       93.91039406948033/(1 + n)^2 - 85.81679567653221/(3 + n) + 
       361.0392247000297*Lm11m1[n] + 232.1144024429168*Lm12m1[n] + 
       35.38568209541474*Lm13m1[n] - (488.477491593376*S[1, n])/n^2) + 
     nf^3*(481.218436212639 - 1.9708872235203123/(1 + n)^3 + 
       5.659666837200436/(1 + n)^2 - 1.6042984530424238/(3 + n) + 
       2.072417794026811*Lm11m1[n] + 1.1315210671628686*Lm12m1[n] + 
       0.14497449792789385*Lm13m1[n] - (1.8446133017881674*S[1, n])/n^2) + 
     nf*(-268.4539014607115 - 1969.0104529610248/(1 + n)^3 - 
       2742.0697059315535/(1 + n)^2 + 2512.6444931763654/(3 + n) - 
       2121.855469704418*Lm11m1[n] - 3590.759053757736*Lm12m1[n] - 
       413.4348940200741*Lm13m1[n] + (13862.898314841788*S[1, n])/n^2)
 
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
 
gqq3nsmN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
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
g3nsm := N[gqq3nsmFitted + gqq3nsmN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nsmFitted = -46765.81834870958 + 194339.87849432364/(1 + n)^3 - 
     88491.39082988899/(1 + n)^2 - 16673.930515791046/(3 + n) - 
     178815.08807753402*Lm11m1[n] - 49111.66200489113*Lm12m1[n] - 
     11804.706477594546*Lm13m1[n] - (103246.60426011353*S[1, n])/n^2 + 
     nf^2*(-387.74515575609433 - 192.0247458490642/(1 + n)^3 + 
       93.91039585555254/(1 + n)^2 - 85.81679545929036/(3 + n) + 
       361.0392268935395*Lm11m1[n] + 232.11440344945356*Lm12m1[n] + 
       35.385682403689145*Lm13m1[n] - (488.4774915260013*S[1, n])/n^2) + 
     nf^3*(-6.030028008641906 - 1.970889300579229/(1 + n)^3 + 
       5.659669709532422/(1 + n)^2 - 1.6042981632010402/(3 + n) + 
       2.072421291623526*Lm11m1[n] + 1.1315226324838452*Lm12m1[n] + 
       0.14497494209496486*Lm13m1[n] - (1.8446131818388185*S[1, n])/n^2) + 
     nf*(11100.227495468602 - 1969.0103560388097/(1 + n)^3 - 
       2742.0698380977074/(1 + n)^2 + 2512.644480640595/(3 + n) - 
       2121.8556302221364*Lm11m1[n] - 3590.759124997347*Lm12m1[n] - 
       413.4349137463167*Lm13m1[n] + (13862.898309133381*S[1, n])/n^2)
 
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
 
gqq3nsmN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
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
g3nsm := N[gqq3nsmFitted + gqq3nsmN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nsmFitted = 21.79452848924608 + 194339.87830907636/(1 + n)^3 - 
     88491.39057190665/(1 + n)^2 - 16673.930488933267/(3 + n) - 
     178815.08776293026*Lm11m1[n] - 49111.66186346875*Lm12m1[n] - 
     11804.70643695953*Lm13m1[n] - (103246.60424951889*S[1, n])/n^2 + 
     nf^2*(-0.035067411788292195 - 192.02474797532508/(1 + n)^3 + 
       93.91039875649527/(1 + n)^2 - 85.81679519012039/(3 + n) + 
       361.03923041518914*Lm11m1[n] + 232.11440501174422*Lm12m1[n] + 
       35.38568283254944*Lm13m1[n] - (488.47749140099427*S[1, n])/n^2) + 
     nf^3*(-0.00006395058927056391 - 1.9708892990266347/(1 + n)^3 + 
       5.659669708128045/(1 + n)^2 - 1.6042981630149458/(3 + n) + 
       2.072421290083369*Lm11m1[n] + 1.131522632034699*Lm12m1[n] + 
       0.14497494216748838*Lm13m1[n] - (1.8446131819724885*S[1, n])/n^2) + 
     nf*(0.13817261057574623 - 1969.0104544141527/(1 + n)^3 - 
       2742.069703040694/(1 + n)^2 + 2512.644493752763/(3 + n) - 
       2121.855466015133*Lm11m1[n] - 3590.7590518401676*Lm12m1[n] - 
       413.43489330330885*Lm13m1[n] + (13862.898314871118*S[1, n])/n^2)
 
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
 
gqq3nsmN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
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
g3nsm := N[gqq3nsmFitted + gqq3nsmN0asy + gqq3nsNinf /. QCDConstantsRules /. 
      ZetaRules]
 
gqq3nsmFitted = 21.79452848924608 + 194339.87830907636/(1 + n)^3 - 
     88491.39057190665/(1 + n)^2 - 16673.930488933267/(3 + n) - 
     178815.08776293026*Lm11m1[n] - 49111.66186346875*Lm12m1[n] - 
     11804.70643695953*Lm13m1[n] - (103246.60424951889*S[1, n])/n^2 + 
     nf^2*(-0.035067411788292195 - 192.02474797532508/(1 + n)^3 + 
       93.91039875649527/(1 + n)^2 - 85.81679519012039/(3 + n) + 
       361.03923041518914*Lm11m1[n] + 232.11440501174422*Lm12m1[n] + 
       35.38568283254944*Lm13m1[n] - (488.47749140099427*S[1, n])/n^2) + 
     nf^3*(-0.00006395058927056391 - 1.9708892990266347/(1 + n)^3 + 
       5.659669708128045/(1 + n)^2 - 1.6042981630149458/(3 + n) + 
       2.072421290083369*Lm11m1[n] + 1.131522632034699*Lm12m1[n] + 
       0.14497494216748838*Lm13m1[n] - (1.8446131819724885*S[1, n])/n^2) + 
     nf*(0.13817261057574623 - 1969.0104544141527/(1 + n)^3 - 
       2742.069703040694/(1 + n)^2 + 2512.644493752763/(3 + n) - 
       2121.855466015133*Lm11m1[n] - 3590.7590518401676*Lm12m1[n] - 
       413.43489330330885*Lm13m1[n] + (13862.898314871118*S[1, n])/n^2)
 
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
 
gqq3nsmN0asy = (-80*cf^4)/n^7 - (160*cf^4)/n^6 - (160*ca*cf^3)/n^5 - 
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
