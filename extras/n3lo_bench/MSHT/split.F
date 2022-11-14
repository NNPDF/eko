* =====================================================================
* =====================================================================
*     N3LO
* =====================================================================
*     
*     This code is provided as supplementary material to the MSHT20 
*     approximate N3LO PDFs.
*     
*     These approximations are not exact and allow for a level of 
*     variation by the input parameter "a" in all cases. These 
*     subroutines here use a simple method of simultaneous equations in
*     order to produce the approximations and therefore may well 
*     be superseded in the future by more robust and equally/more 
*     efficient approximations or indeed, the exact functions when 
*     available.
*
*     Please refer to arXiv:2207.XXXX for the full details of how these
*     approximations are determined.
*
*     Note that these can and will be continually updated with any  
*     extra exact calculations that are made available, which will 
*     increase the accuracy of these approximations.
*
*     When using these approximations, please reference the following.
*
*     References:
*     J. McGowan, et. al., arXiv:2207.XXXXXX
*     S. Moch, et. al., arXiv:1707.08315
*     A. Vogt, et. al., arXiv:1808.08981
*     S. Catani, et. al., hep-ph/9405388
*     L. N. Lipatov, Sov. J. Nucl. Phys. (1976)
*     E. A. Kuraev, et. al., JETP (1977)
*     I. I. Balitsky, et. al., Sov. J. Nucl. Phys. (1978)
*     V. S. Fadin, et. al., hep-ph/9802290
*     M. Ciafaloni, et. al., hep-ph/9803389
*     
*     
* ---------------------------------------------------------------------
*     P_{qq}^{NS}
* ---------------------------------------------------------------------
*
*     Contribution from approximate four-loop splitting function with
*     variational parameter (0 < aNS < 0.014). This is a subroutine and
*     outputs a regular and plus distribution part of the approximation.
*
*     J. M. - 11/07/2022
*
      SUBROUTINE Pqqns3PAB(x, a, FLAVOR, PnsN3LOresult, PnsN3LOplus)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      INTEGER n_dimension
      DOUBLE PRECISION N(8), C(8), N_exact(8), N_a(8), inv_A(8, 8)
      DATA PI,PI2/3.14159,9.8696/
*
      n_dimension=8
* 
*     Moments for the known exact small-x and large-x contributions (Vogt)
      N_exact(1) = -0.00021211D0 - FLAVOR*(0.1654801313006386D0 + 0.20740623526104135D0
     .-0.1502823524D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.004654314166031352D0 -0.007842937752425193D0
     .+ 4.7723213922D-03 + 7.774039377D-03) 
     .- FLAVOR**3.*(-6.62104090137594D-05 - 0.0001312258806063389D0 
     .+ 7.653104787D-05 + 1.209052801D-04)

      N_exact(2) = -0.00021141D0 - FLAVOR*(0.11120681844515805D0 + 0.3802447646452425D0 
     .-0.2688473229D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(9.571697043e-3 + 7.774039377D-03 -0.0029670140315604917D0
     .-0.014378719212779521D0) 
     .- FLAVOR**3.*(-3.609651149299798D-05 -0.00024058078111162135D0 
     .+ 1.55772043D-04 + 1.209052801D-04)

      N_exact(3) = -0.00021069D0 - FLAVOR*(0.08533032993183774D0 + 0.4735775705127111D0 
     .-0.3363034306D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(0.01237066021 + 7.774039377D-03 
     .- 0.002236654071582081D0 -0.01790804120137086D0) 
     .- FLAVOR**3.*(-2.4727034394311423D-05 -0.0002996324273844739D0 
     .+ 2.034542201D-04 + 1.209052801D-04)

      N_exact(4) = -0.00020986D0 - FLAVOR*(0.06978655862237221D0 + 0.5377747385697003D0
     .-0.3849568093D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.001806806533621782D0 -0.020335617172359613D0 
     .+ 0.01436839622 + 7.774039377D-03) 
     .- FLAVOR**3.*(-1.875753442493042D-05 -0.00034024996185786445D0 
     .+ 2.381022604D-04 + 1.209052801D-04)

      N_exact(5) = -0.00020809D0 - FLAVOR*(0.059311724466073266D0 + 0.5867456552285572D0 
     .-0.4234530704D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.0015200868207649136D0 -0.02218742191946D0 
     .+ 0.01593349271 + 7.774039377D-03) 
     .- FLAVOR**3.*(-1.5092063174265756D-05 - 0.0003712338503343612 
     .+ 2.654206822D-04 + 1.209052801D-04)

      N_exact(6) = -0.0002073D0 - FLAVOR*(0.0517320184770643D0 + 0.6263413910511195D0 
     .-0.4554694245D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.001314205205204153D0 -0.023684710035832085D0 
     .+ 0.0172249122 + 7.774039377D-03) 
     .- FLAVOR**3.*(-1.2617371403920047D-05 -0.0003962860639046623D0 
     .+ 2.879982076D-04 + 1.209052801D-04)

      N_exact(7) = -0.00020562D0 - FLAVOR*(0.045972268757607376D0 + 0.6595795697788507D0 
     .-0.4829482701D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.0011588022146498728D0 -0.024941591085900227D0 
     .+ 0.0183264033 + 7.774039377D-03) 
     .- FLAVOR**3.*(-1.0836262424080888D-05 -0.0004173158524633704D0 + 3.072468902D-04 + 1.209052801D-04)

      N_exact(8) = -0.00020381D0 - FLAVOR*(0.04143575048971359D0 + 0.6882213832196611D0 
     .-0.5070540331D0 + 0.01076711056D0 - 0.2333500675D0) 
     .- FLAVOR**2.*(-0.0010371493846056854D0 -0.026024663442187512D0 
     .+ 0.01928783511D0 + 7.774039377D-03) 
     .- FLAVOR**3.*(-9.493881167514427D-06 -0.00043543752168996016D0 
     .+ 3.240261808D-04 + 1.209052801D-04)
* 
*     Moments for the varied function multiplied by a (ln^3(x))
      N_a(1) = -0.375D0
      N_a(2) = -0.0234375D0
      N_a(3) = -0.004629629629528962D0
      N_a(4) = -0.0014648437499999065D0
      N_a(5) = -0.0005999999999999996D0
      N_a(6) = -0.0002893518518518518D0
      N_a(7) = -0.00015618492294877136D0
      N_a(8) = -9.155273437499997D-05
*      
      DO i=1, n_dimension
        N(i) = N_exact(i) - a * N_a(i)
      ENDDO
*
      inv_A = reshape((/9.67452285D-02, 7.12205880D03, 2.55568773D03, 
     &                  4.41612871D02, -3.63251998D02, 8.54599881D03, 
     &                  -8.03082218D03, 5.66780493D01, -1.10377154D01,
     &                  -7.11199538D05, -2.58657431D05, -4.59602884D04,  
     &                  2.63108547D04, -8.24798082D05, 7.81924565D05,  
     &                  -3.25280541D03, 1.99489100D02, 1.15489637D07,  
     &                  4.23773320D06, 7.69318756D05, -3.54975874D05,  
     &                  1.31604354D07, -1.25178270D07, 4.09448382D04,  
     &                  -1.28368247D03, -6.79475656D07, -2.50856870D07,  
     &                  -4.63268153D06, 1.85137884D06, -7.66032630D07,  
     &                  7.29666144D07, -2.06388099D05, 3.82370653D03,  
     &                  1.87306734D08, 6.94570801D07, 1.30090317D07,  
     &                  -4.68326682D06, 2.09632375D08, -1.99805999D08,  
     &                  5.11458122D05, -5.75484319D03, -2.63212244D08,  
     &                  -9.79232841D07, -1.85597818D07, 6.16625773D06,  
     &                  -2.93018936D08, 2.79351504D08, -6.64139088D05,  
     &                  4.25513991D03, 1.82949771D08, 6.82322632D07,  
     &                  1.30646558D07, -4.07104678D06, 2.02835925D08,  
     &                  -1.93381943D08, 4.34087442D05, -1.22905233D03,  
     &                  -4.99408525D07, -1.86617424D07, -3.60495568D06,  
     &                  1.06569809D06, -5.51895151D07, 5.26130529D07,  
     &                  -1.12766384D05/), shape=(/8,8/)) 
* 
*     Matrix multiplication of inv_A and N
*
      DO i=1,n_dimension
        C(i) = 0.D0
        DO j=1,n_dimension
          C(i) = C(i) + inv_A(i, j) * N(j)
        ENDDO
      ENDDO
*
      PnsN3LOplus = (C(1) * (1./(1.-X))) * (4*PI)**4.
      PnsN3LOresult = (C(2) * (1.-X) * LOG(1.-X) + C(3) * (1.-X) 
     &          * (LOG(1.-X)**2.) + C(4) * (1.-X) * (LOG(1.-X)**3.) 
     &          + C(5) + C(6) * X + C(7) * (X**2.) + C(8) * (LOG(X)**2.)
     &          + (a) * (LOG(X)**3.)) * (4*PI)**4.

*
      RETURN
      END
*
* -------------------------------------------------------------------------
*
*     This is the regular exact part of P3NS.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqqns3PexactA(x, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI,PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      Pqqns3PexactA = (19069.80D0 * AL1 - 29733.85D0 + 25000D0*((1 - x) *
     &          (3.5242 + 8.3679D0 * x - 1.2395D0 * (x**2.) + 1.7423D0 * 
     &          (x**3.)) + 11.916D0 * x * log(x) - 0.2237D0 * x * 
     &          (log(x)**2.) - 0.0129D0 * x * (log(x)**3.) +11.937D0 * 
     &          (1. - x) * AL1 + 13.955D0 * log(x) * AL1) + 51671.33D0 * 
     &          log(x)+17120.95D0*(log(x)**2.)+2863.226D0 * (log(x)**3.) 
     &          + 297.8255D0*(log(x)**4.)+16.D0*(log(x)**5.)+(1.D0/2.D0) 
     &          * (log(x)**6.) + FLAVOR * (- 3079.761D0 * AL1 + 8115.605D0 
     &          + 2500.D0 * ((1. - x) * (-7.4077D0+4.5141D0*x-1.0069D0 * 
     &          (x**2.) + 0.7641D0 * (x**3.)) + 8.4211D0 * (1 - x) * AL1 
     &          + 7.5633D0*log(x)*AL1 + 7.5236D0 * x * log(x)+0.2208D0 
     &          * x * (log(x)**2.) + 0.05712D0 * x * (log(x)**3.)) 
     &          - 9239.374D0*log(x)-2917.312D0*(log(x)**2.)-430.5308D0 
     &          *(log(x)**3.)- 36.D0*(log(x)**4.) -(4.D0 /3.D0) * 
     &          (log(x)**5.)) + (FLAVOR**2.) * (26.68861D0 *AL1 - 376.0092D0 
     &          + 250.D0*((1. - x)*(3.0008D0 + 0.8619D0*x - 0.12411D0 * 
     &          (x**2.) +0.31595D0 * (x**3.)) - 0.37529D0 * x * log(x) 
     &          - 0.21684D0*x*(log(x)**2.) -0.02295D0 * x * (log(x)**3.)
     &          + (0.03394D0 * (1. - x) + 0.40431D0 * log(x)) * AL1) 
     &          + 393.0056D0 * log(x) + 112.5705D0 * (log(x)**2.) 
     &          + 16.52675D0 * (log(x)**3.)+0.7901235D0 * (log(x)**4.)) 
     &          + (FLAVOR**3.) * (-2.426296D0-0.8460488D0* x+(0.5267490D0* 
     &          (1./(1.-x)) - 3.687243D0+3.160494D0 * x) * log(x) -
     &          (0.1316872D0 * (10.*(1./(1. - x)) + 1.)-1.448560D0 * x) 
     &          * (log(x)**2.) - (0.2633745D0 *(1./(1.-x))-0.1316870D0 
     &          * (1.+x)) * (log(x)**3.)) - 507.152D0 * (- 1.)-1777.27D0 
     &          * (AL1**2) * (1 - x) - 204.183D0 * (AL1**3.)*(1.-x) + 
     &          3948.16D0 * (1.-x) - 2464.61D0* (2.*x - (x**2.)) 
     &          * (1.-x) - 1839.44D0* (log(x)**2.) - 402.156D0 * 
     &          (log(x)**3.) - 55.87553D0 * (log(x)**4.) - 2.831276D0
     &          * (log(x)**5.) - 0.1488340D0 * (log(x)**6.) - 2601.749D0 
     &          - 2118.867D0 * AL1 + FLAVOR * (7.33927D0 * (-1.) - 143.813D0 
     &          * AL1 * (1. - x) - 18.8803D0 *(AL1**3.) * (1. - x) 
     &          - (1116.34D0 - 1071.24D0 * x) * x * (1. - x)- 59.3041D0 
     &          * (log(x)**2.) - 8.4620D0* (log(x)**3.) + 4.658436D0 * (
     &          log(x)**4.) + 0.2798354D0 * (log(x)**5.) + 312.1643D0 
     &          + 337.9310D0 * AL1))
*
      RETURN
      END
*
* ---------------------------------------------------------------------------
*
*     This is the singular exact part of P3NS.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqqns3PexactB(x, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI,PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      Pqqns3PexactB = (21209.02D0 * (1. / (1. - x)) + FLAVOR * (
     &                   -5179.372D0 * (1. / (1. - x))) + (FLAVOR**2.) * 
     &                   (195.5772D0 * (1. / (1. - x))) + (FLAVOR**3.) * 
     &                   (3.272344D0 * (1. / (1. - x))) + (- 507.152D0 
     &                   + FLAVOR * 7.33927D0) * (1. / (1. - x)))
*
      RETURN
      END
*
* ----------------------------------------------------------------------------
*
*     This is the 'local' exact part of P3NS.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqqns3PexactC(X, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI, PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      PnsN3LOEXACTCONST = (-2405.03D0+FLAVOR*267.965D0+FLAVOR**2.*(193.8554D0
     &      +0.0037D0)+FLAVOR**3.*(3.014982D0)+(25796.09D0-1.0D0))
*
      Pqqns3PexactC = PnsN3LOEXACTCONST - (21209.02D0*(AL1) + FLAVOR * (
     &                   -5179.372D0 * AL1) + (FLAVOR**2.) * 
     &                   (195.5772D0 * AL1) + (FLAVOR**3.) * 
     &                   (3.272344D0 * AL1) + (- 507.152D0 
     &                   + FLAVOR * 7.33927D0) * AL1)
*
      RETURN
      END
*
* ---------------------------------------------------------------------
*
*
* ..This is the regular piece of P3_NS-. Approximation taken from 
*   arXiv:1707.08315.
*
*     J. M. - 11/07/2022
*
      FUNCTION P3NSMA (Y, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
*
      DL  = LOG (Y)
      DL1 = LOG (1.-Y)
      D81 = 1./81.D0

      PLNFminusA = FLAVOR**2. * (26.68861*DL1 - 376.0092 
     ,            + 250.0*((1.-y)*(3.2206 
     ,            + 1.7507*y + 0.13281*y**2. + 0.45969*y**3.)
     ,            + 1.5641*y*DL - 0.37902*y*DL**2. - 0.03248*y*DL**3.
     ,            + (2.7511*(1.-y) + 3.2709*DL)*DL1) + 437.8810*DL 
     ,            + 128.2948*DL**2. + 19.59945*DL**3. + 0.9876543*DL**4.)
     ,            + FLAVOR**3.*(- 2.426296
     ,            - 0.8460488*y + (0.5267490*(1./(1.-y)) - 3.687243 
     ,            + 3.160494*y)*DL - (0.1316872*(10.*(1./(1.-y)) + 1.0)
     ,            - 1.448560*y)*DL**2. - (0.2633745*(1./(1.-y)) 
     ,            - 0.1316870*(1.+y))*DL**3.)

      PL0_PL1A = 19069.80D0 * DL1 - 29733.85D0 + 25000D0*((1. - y) *
     ,          (3.5242 + 8.3679D0 * y - 1.2395D0 * (y**2.) + 1.7423D0 * 
     ,          (y**3.)) + 11.916D0 * y * log(y) - 0.2237D0 * y * 
     ,          (log(y)**2.) - 0.0129D0 * y * (log(y)**3.) +11.937D0 * 
     ,          (1. - y) * DL1 + 13.955D0 * log(y) * DL1) + 51671.33D0 * 
     ,          log(y)+17120.95D0*(log(y)**2.)+2863.226D0 * (log(y)**3.) 
     ,          + 297.8255D0*(log(y)**4.)+16.D0*(log(y)**5.)+(1.D0/2.D0) 
     ,          * (log(y)**6.) + FLAVOR * (- 3079.761D0 * DL1 + 8115.605D0 
     ,          + 2500.D0 * ((1. - y) * (-7.4077D0+4.5141D0*y-1.0069D0 * 
     ,          (y**2.) + 0.7641D0 * (y**3.)) + 8.4211D0 * (1. - y) * DL1 
     ,          + 7.5633D0*log(y)*DL1 + 7.5236D0 * y * log(y)+0.2208D0 
     ,          * y * (log(y)**2.) + 0.05712D0 * y * (log(y)**3.)) 
     ,          - 9239.374D0*log(y)-2917.312D0*(log(y)**2.)-430.5308D0 
     ,          *(log(y)**3.)- 36.D0*(log(y)**4.) -(4.D0 /3.D0) * 
     ,          (log(y)**5.))
*
      P3NSMA =   -502.481*(- 1.)
     ,            - 3997.39*DL1*(1.-y) + 511.567*DL1**3.*(1.-y)
     ,            + (4043.59 - 15386.6*y)*y*(1.-y) + 1532.96*DL**2.
     ,            + 31.6023*DL**3. + 0.4964335*(DL**6. + 18*DL**5.)
     ,            - 2601.749 - 2118.867*DL1 + FLAVOR*(7.82077*(
     ,            -1.0) - 1360.04*DL1*(1.-y) 
     ,            + 38.7337*DL1**3.*(1.-y) - (335.995*(2.+y) 
     ,            + 1605.91*y**2)*(1.-y) - 9.76627*DL**2. + 0.14218*DL**5.
     ,            + 312.1643 + 337.9310*DL1) + PL0_PL1A + PLNFminusA
*
      RETURN
      END
*
* ---------------------------------------------------------------------
*
*
* ..This is the singular piece of P3_NS-. Approximation taken from 
*   arXiv:1707.08315.
*
*     J. M. - 11/07/2022
*
      FUNCTION P3NSMB (Y, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
*
      DL  = LOG (Y)
      DL1 = LOG (1.-Y)
      D81 = 1./81.D0

      PLNFminusB = FLAVOR**2. * (195.5772*(1./(1.-y)))
     ,            + FLAVOR**3.*(3.272344*(1./(1.-y)))

      PL0_PL1B = 21209.02*(1./(1.-y)) + FLAVOR*(-5179.372*(1./(1.-y)))
*
      P3NSMB =   -502.481*((1./(1.-y))) + FLAVOR*(7.82077*(1./(1.-y)))
     ,            + PLNFminusB + PL0_PL1B
*
      RETURN
      END
*
* ---------------------------------------------------------------------
*
*
* ..This is the 'local' piece of P3_NS-. Approximation taken from 
*   arXiv:1707.08315.
*
*     J. M. - 11/07/2022
*
      FUNCTION P3NSMC (Y, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
*
      DL  = LOG (Y)
      DL1 = LOG (1.-Y)
      D81 = 1./81.D0

      PLNFminusC = FLAVOR**2. * ((193.8554 + 0.0037)) 
     ,            + FLAVOR**3.*(3.014982)

      PL0_PL1C = (25796.09 - 1.0) - FLAVOR*(5818.637 + 0.35)

      P3NSMresidual = - (21209.02D0*(DL1) + FLAVOR * (
     &                   -5179.372D0 * DL1) + (FLAVOR**2.) * 
     &                   (195.5772D0 * DL1) + (FLAVOR**3.) * 
     &                   (3.272344D0 * DL1) + (- 507.152D0 
     &                   + FLAVOR * 7.33927D0) * DL1)
*
      P3NSMC =   - 2380.255 + FLAVOR*(270.468) + PL0_PL1C + PLNFminusC
     ,            + P3NSMresidual
*
      RETURN
      END
*
* ---------------------------------------------------------------------
*
*
* ..This is P3_NSS, the difference of P3_NSV and P3_NS-.  
*   Approximation taken from arXiv:1707.08315.
*
*     J. M. - 11/07/2022
*
      FUNCTION P3NSSA (Y, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
*
      DL  = LOG (Y)
      DL1 = LOG (1.-Y)
      D81 = 1./81.D0

      P3NSSA = FLAVOR * (60.40*(1.-y)*DL1**2. + 4.685*(1.-y)*DL1**3. 
     ,            + (1.-y)*y*(4989.2 - 1607.73*y) + 3687.6*DL 
     ,            + 3296.6*DL**2. + 1271.11*DL**3. + 533.44*DL**4.
     ,            + 97.27*DL**5. + 4.*DL**6.)
     ,            + FLAVOR**2. * (19.70002*(1.-y)*DL1 
     ,            - 3.435474*(1.-y)*DL1**2. + 250.0*((1.-y)*(
     ,            -4.7656 + 1.6908*y + 0.1703*y**2.) - 0.41652*y*DL
     ,            + 0.90777*y*DL**2. + 0.12478*y*DL**3. 
     ,            + 0.17155*(1.-y)*DL1 + 0.17191*DL*DL1) - 647.3971*DL
     ,            - 66.41219*DL**2. - 5.353347*DL**3. - 5.925926*DL**4.
     ,            - 0.3950617*DL**5.)
*
      RETURN
      END
*
* ---------------------------------------------------------------------
*
*     This part collects the regular pieces for the desired non-singlet
*     distribution (+, - or sea).
*
*     J. M. - 11/07/2022
*
*
      FUNCTION P3NSA(y, aNS, ipm, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      ipm2=ipm*ipm
      p3nsa = 0.d0

      CALL Pqqns3PAB(y, aNS, FLAVOR, p3nspaa, p3nspba)

      if(ipm2.ne.1.and.ipm.ne.-2) go to 1 ! G.W. 04/07/2007

      if(ipm.eq.1) p3nsa = p3nspaa + Pqqns3PexactA(y, FLAVOR)
      if(ipm.eq.-1) p3nsa = P3NSMA(y,FLAVOR)
      if(ipm.eq.-2) p3nsa = P3NSMA(y,FLAVOR) + P3NSSA (Y, 2.0D0)
      return
*
    1 print 99,ipm
      stop
   99 format(1x,'ipm=',i3)
*
      END
*
* ---------------------------------------------------------------------
*
*     This part collects the singular pieces for the desired non-singlet
*     distribution (+, - or sea).
*
*     J. M. - 11/07/2022
*
      FUNCTION P3NSB(y, aNS, ipm, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      ipm2=ipm*ipm
      p3nsb = 0.d0

      CALL Pqqns3PAB(y, aNS, FLAVOR, p3nspaa, p3nspba)

      if(ipm2.ne.1.and.ipm.ne.-2) go to 1 ! G.W. 04/07/2007

      if(ipm.eq.1) p3nsb = p3nspba + Pqqns3PexactB(y, FLAVOR)
      if(ipm.eq.-1) p3nsb = P3NSMB(y,FLAVOR)
      if(ipm.eq.-2) p3nsb = P3NSMB(y,FLAVOR) ! p2nsma(y,nf) + p2nssa(y,2) ! G.W. 05/07/2007
      return
*
    1 print 99,ipm
      stop
   99 format(1x,'ipm=',i3)
*
      END
*
* ---------------------------------------------------------------------
*
*     This part collects the 'local' pieces for the desired non-singlet
*     distribution (+, - or sea).
*
*     J. M. - 11/07/2022
*
*
      FUNCTION P3NSC(y, aNS, ipm, FLAVOR)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      ipm2=ipm*ipm
      p3nsc = 0.d0              ! G.W. 07/05/2007
      AL1 = LOG(1.-y)

      CALL Pqqns3PAB(y, aNS, FLAVOR, p3nspaa, p3nspba)
*
      if(ipm2.ne.1.and.ipm.ne.-2) go to 1 ! G.W. 04/07/2007
*
      if(ipm.eq.1) p3nsc = Pqqns3PexactC(y, FLAVOR) + p3nspba*(1.-y)*AL1
      if(ipm.eq.-1) p3nsc = P3NSMC(y,FLAVOR) ! p2nsmc(y,nf)
      if(ipm.eq.-2) p3nsc = P3NSMC(y,FLAVOR) ! p2nsmc(y,nf) ! G.W. 04/07/2007
      return
*
    1 print 99,ipm
      stop
   99 format(1x,'ipm=',i3)
*
      END
*
* =====================================================================
*     P_{qq}^{PS}
! ---------------------------------------------------------------------
*     
*     Contribution from approximate four-loop splitting function with
*     variational parameter (-0.7 < aPS < 0.0).
*     This is the regular part of P3PS.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqqps3A(x, a)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      INTEGER n_dimension
      DOUBLE PRECISION N(4), C(4),  
     &        N_exact(4), N_a(4)
      DOUBLE PRECISION inv_A(4, 4)
      DATA PI,PI2/3.14159,9.8696/
*
      n_dimension=4
* 
*     Moments for the known exact small-x contribution (Vogt)
      N_exact(1) = -1.28827235D-01
      N_exact(2) = -2.42108771D-03
      N_exact(3) = -6.81270332D-05
      N_exact(4) = 4.79640375e-05
* 
*     Moments for the varied function multiplied by a (ln(1/x)/x)*(1-x)
      N_a(1) = 0.75D0
      N_a(2) = 0.04861111111078488D0
      N_a(3) = 0.012222222222363594D0
      N_a(4) = 0.004783163265306247D0
*
      DO i=1, n_dimension
        N(i) = N_exact(i) - a * N_a(i)
      ENDDO
*
      inv_A = reshape((/-4.33423515D+01, 1.04394869D+03, 
     &                  -3.26481918D+03, 2.49215628D+03, 1.01800794D+02, 
     &                  -2.29059693D+03, 7.08126548D+03, 
     &                  -5.37933607D+03, 1.28895321D+02, 
     &                  -3.28605015D+03, 1.09724714D+04, 
     &                  -8.75612151D+03, -9.56650450D0, 2.57515782D+02, 
     &                  -9.20800163D+02, 7.84949427D+02/), 
     &                 shape=(/4,4/))

* 
*     Matrix multiplication of inv_A and N
*
      DO i=1,n_dimension
        C(i) = 0.D0
        DO j=1,n_dimension
          C(i) = C(i) + inv_A(j, i) * N(j)
        ENDDO
      ENDDO
*
      Pqqps3A = ((1.-X) * (C(1) * (1./X) + C(2) * (LOG(X)**2.) 
     &         + C(3) * (X**2.) + C(4) * (LOG(1.-X)**2.) 
     &         + a * LOG(1./X)/X) + Pqqps3exactA(X)) * (4*PI)**4.
*
      RETURN
      END
*
* ------------------------------------------------------------------------
*
*     This is the regular exact part of P3PS.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqqps3exactA(x)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI,PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      Pqqps3exactA = (((CA ** 2.0)*CF / (6.0D0 * (PI**4.))) * (
     &                (82.0D0/81.0D0) + 2.0 * 1.20206D0) 
     &                * (log(1.0 / x)**2.) / x)
*
      RETURN
      END
*
* =====================================================================
*     P_{qg}
! ---------------------------------------------------------------------
*
*     Contribution from approximate four-loop splitting function with
*     variational parameter (-2.5 < aQG < -0.8).
*     This is the regular part of P3QG.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqg3A(x, a)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      INTEGER n_dimension
      DOUBLE PRECISION N(4), C(4), N_exact(4), 
     &        N_a(4)
      DOUBLE PRECISION inv_A(4, 4)
      DATA PI,PI2/3.14159,9.8696/
*
      n_dimension=4
* 
*     Moments for the known exact small-x contribution (Vogt)
      N_exact(1) = -0.32821791D0
      N_exact(2) = -0.0079068D0
      N_exact(3) = 0.00668085D0
      N_exact(4) = 0.0112976D0
* 
*     Moments for the varied function multiplied by a (ln(1/x)/x)
      N_a(1) = 0.9999999999999999D0
      N_a(2) = 0.11111111111076043D0
      N_a(3) = 0.04000000000013782D0
      N_a(4) = 0.02040816D0
*
      DO i=1, n_dimension
        N(i) = N_exact(i) - a * N_a(i)
      ENDDO
*
      inv_A = reshape((/1.50859345D1, -1.73502577D2, 3.81181701D2, -2.25021001D2,
     &                  1.39350535D1, -2.33166358D2, 5.53133581D2, -3.37644014D2,
     &                  3.92339377D0, -8.52278788D1, 2.57493287D2, -1.79628628D2,
     &                  -1.15513419D-2, 2.74882781D-1, -9.19800075D-1, 7.20490480D-1/),
     &                  shape=(/4,4/))
*
*     Matrix multiplication of inv_A and N
*
      DO i=1,n_dimension
        C(i) = 0.D0
        DO j=1,n_dimension
          C(i) = C(i) + inv_A(j, i) * N(j)
        ENDDO
      ENDDO
*
      Pqg3A = (C(1) * LOG(X)**2. + C(2) * LOG(X) + C(3) * (X**2.) 
     &          + C(4) * LOG(1.-X)**4.
     &          + a * LOG(1./X)/X + Pqg3exactA(X)) * (4*PI)**4.
*
      RETURN
      END
*
* ---------------------------------------------------------------------------
*
*     This is the regular exact part of P3QG.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pqg3exactA(x)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
*
      DATA PI,PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      Pqg3exactA = ((CA ** 3.0)/ (6.0 * (PI**4.))) * 
     &               ((82.0D0 / 81.0D0) + 2.0 * 1.20206D0) * 
     &               (log(1.0 / x)**2.) / x
*
      RETURN
      END
*
* =====================================================================
*     P_{gq}
! ---------------------------------------------------------------------
*
*     Contribution from approximate four-loop splitting function with
*     variational parameter (-1.8 < aGQ < -1.5).
*     This is the regular part of P3GQ.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pgq3A(x, a)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      INTEGER n_dimension
      DOUBLE PRECISION N(4), C(4), N_exact(4), 
     &        N_a(4)
      DOUBLE PRECISION inv_A(4, 4)
      DATA PI,PI2/3.14159,9.8696/
*
      n_dimension=4
*
* 
*     Moments for the known exact small-x contribution (Vogt)
      N_exact(1) = -0.78602265D0
      N_exact(2) = 0.05776039D0
      N_exact(3) = 0.04868481D0
      N_exact(4) = 0.04071129D0
* 
*     Moments for the varied function multiplied by a (ln^2(1/x)/x)
      N_a(1) = 1.9999999999999991D0
      N_a(2) = 0.07407407407348075D0
      N_a(3) = 0.016000000001040217D0
      N_a(4) = 0.0058309037900882415D0
*
      DO i=1, n_dimension
        N(i) = N_exact(i) - a * N_a(i)
      ENDDO
*
      inv_A = reshape((/-15.48054909D0, -39.22467343D0, 13.63767859D0, 
     &                 3.69946174D0, 442.68554891D0, 1038.70260338D0, 
     &                 -415.55730182D0,-113.79603728D0,-1263.91174116D0, 
     &                 -2938.1594667D0, 1302.32965951D0,362.67726047D0, 
     &                 874.63704072D0, 2025.64293423D0,-958.33374045D0, 
     &                 -272.56485302D0/), shape=(/4,4/))
*
*     Matrix multiplication of inv_A and N
      DO i=1,n_dimension
        C(i) = 0.D0
        DO j=1,n_dimension
          C(i) = C(i) + inv_A(i, j) * N(j)
        ENDDO
      ENDDO
*
      Pgq3A = (C(1) * LOG(1./X)/X + C(2) * (LOG(X)**3.) + C(3) * X 
     &          + C(4) * LOG(1.-X) + a * (LOG(1./X)**2.)/X 
     &          + Pgq3exactA(X)) * (4*PI)**4.
*
      RETURN
      END
*
* ---------------------------------------------------------------------------
*
*     This is the regular exact part of P3GQ.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pgq3exactA(x)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI,PI2/3.14159,9.8696/
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
*
      Pgq3exactA = ((CA ** 3.0) * CF / (3.0 * (PI**4.))) * 1.20206D0 
     &               * (log(1.0 / x)**3.) / x
*
      RETURN
      END
*
* =====================================================================
*     P_{gg}
! ---------------------------------------------------------------------
*
*     Contribution from approximate four-loop splitting function with
*     variational parameter (-5 < aGG < -15).
*     This is the regular part of P3GG.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pgg3A(x, a)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      INTEGER n_dimension
      DOUBLE PRECISION N(4), C(4), N_exact(4), 
     &        N_a(4), inv_A(4, 4)
      DATA PI,PI2/3.14159,9.8696/
*
      n_dimension=4
* 
*     Moments for the known exact small-x contribution (Vogt)
      N_exact(1) = 6.94542399D0
      N_exact(2) = 0.01255781D0
      N_exact(3) = -0.24160401D0
      N_exact(4) = -0.27416992D0
* 
*     Moments for the varied function multiplied by a (ln(1/x)/x)
      N_a(1) = 0.9999999999999999D0
      N_a(2) = 0.11111111111076043D0
      N_a(3) = 0.04000000000013782D0
      N_a(4) = 0.02040816D0
*
      DO i=1, n_dimension
        N(i) = N_exact(i) - a * N_a(i)
      ENDDO
*
      inv_A = reshape((/1.57592805D1, -1.89525930D2, 4.34798303D2, -2.67019531D2,
     &                  1.51046749D1, -2.60999383D2, 6.46267166D2, -4.10596673D2,
     &                  6.05043429D0, -1.35844233D2, 4.26863395D2, -3.12298274D2,
     &                  -3.86434270D-1, 9.19582568D0, -3.07706474D1, 2.41030189D1/),
     &                  shape=(/4,4/))
*
*     Matrix multiplication of inv_A and N     
      DO i=1,n_dimension
        C(i) = 0.D0
        DO j=1,n_dimension
          C(i) = C(i) + inv_A(j, i) * N(j)
        ENDDO
      ENDDO
*
      Pgg3A = (C(1) * LOG(X)**2. + C(2) * LOG(X) + C(3) * (X**2.) 
     &          + C(4) * (LOG(1.-X)**2.) 
     &          + a * LOG(1./X)/X + Pgg3exactA(X)) * (4*PI)**4.
*
      RETURN
      END
*
* -----------------------------------------------------------------------------
*
*     This is the regular exact part of P3GG.
*
*     J. M. - 11/07/2022
*
      FUNCTION Pgg3exactA(x)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DATA PI,PI2/3.14159,9.8696/
      DOUBLE PRECISION NF4
*
      CA = 3.0D0
      CF = 4.0D0/3.0D0
      AL1 = LOG(1.-X)
      NF4 = 4.0D0 ! Set Nf = 4 since the moments in the approximation 
                  ! are for 4 flavours.
*
      Pgg3exactA = ((CA ** 4.0D0) / (3.0D0 * (PI**4.0D0))) * 1.20206D0 
     &               * (log(1.0D0/x)**3.0D0)/ x + (1.0D0 / ((PI**4.0D0))) 
     &               * ((CA**4.0D0)*(-(1205.0D0/162.0D0)+(67.0D0/36.0D0) 
     &               * ((PI**2.0)/6.0D0)+(1.0D0/4.0D0) * (((PI**2.0) 
     &               / 6.0D0)**2.0)-(11.0D0/2.0D0)*1.20206D0) + NF4 
     &               * (CA ** 3.0) * (-(233.0D0/162.0D0)+(13.0D0/36.0D0) 
     &               * ((PI**2.0)/6.0D0)-(1.0D0/3.0D0) * 1.20206D0) + NF4 
     &               * (CA ** 2.0) * CF * ((617.0D0/243.0D0) 
     &               - (13.0D0/18.0D0)*((PI**2.0)/6.0D0) + (2.0D0/3.0D0) 
     &               * 1.20206D0))*(1.0D0/2.0D0)*((log(1.0/x)**2.0)/x)
*
      RETURN
      END
*
