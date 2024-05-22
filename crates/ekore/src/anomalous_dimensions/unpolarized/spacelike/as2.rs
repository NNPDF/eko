//! NLO QCD

use::num::complex::Complex;
use std::f64::consts::LN_2;

use crate::constants::{CA, CF, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};


/// Compute the valence-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.6) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsm(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp1m = c.get(K::S1mh);
    let Sp2m = c.get(K::S2mh);
    let Sp3m = c.get(K::S3mh);
    let g3n = c.get(K::G3);

    let gqq1m_cfca = 16.0 * g3n - (144.0  + N * (1.0 + N) * (156.0 + N * (340.0 + N * (655.0 + 51.0 * N * (2.0 + N)))))/(18.0 * N.powu(3)*(1. + N).powu(3)) + (-14.666666666666666 + 8.0 / N - 8.0 / (1.0 + N))*S2 - (4.0 * Sp2m)/(N + N.powu(2)) + S1*(29.77777777777778 + 16.0 / N.powu(2) - 16.0 * S2 + 8.0 * Sp2m) + 2.0 * Sp3m + 10.0 * ZETA3 + ZETA2 * (16.0 * S1 - 16.0 * Sp1m - (16.0 * (1. + N * LN_2))/N);
    let gqq1m_cfcf = -32. * g3n + (24. - N * (-32. + 3. * N * (-8. + N * (3. + N) * (3. + N.powu(2)))))/(2. * N.powu(3) * (1. + N).powu(3)) + (12. - 8. / N + 8./(1. + N)) * S2 + S1 * (-24. / N.powu(2) - 8./(1. + N).powu(2) + 16. * S2 - 16. * Sp2m) + (8. * Sp2m)/(N + N.powu(2)) - 4. * Sp3m - 20. * ZETA3 + ZETA2 * (-32. * S1 + 32. * Sp1m + 32. * (1. / N + LN_2));
    let gqq1m_cfnf = (-12. + N * (20. + N * (47. + 3. * N * (2. + N))))/(9. * N.powu(2) * (1. + N).powu(2)) - (40. * S1)/9. + (8. * S2)/3.;
    
    CF * ( CA * gqq1m_cfca + CF * gqq1m_cfcf + 2.0 * TR * (nf as f64) * gqq1m_cfnf )
}


/// Compute the singlet-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsp(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp1p = c.get(K::S1h);
    let Sp2p = c.get(K::S2h);
    let Sp3p = c.get(K::S3h);
    let g3n = c.get(K::G3);

    let gqq1p_cfca = -16. * g3n + (132. - N * (340. + N * (655. + 51. * N * (2. + N)))) / (18. * N.powu(2) * (1. + N).powu(2)) + (-14.666666666666666 + 8./N - 8./(1. + N)) * S2 - (4.*Sp2p)/(N + N.powu(2)) + S1*(29.77777777777778 - 16./N.powu(2) - 16. * S2 + 8. * Sp2p) + 2. * Sp3p + 10. * ZETA3 + ZETA2 * (16. * S1 - 16. * Sp1p + 16. * (1./N - LN_2));
    let gqq1p_cfcf = 32. * g3n - (8. + N * (32. + N * (40. + 3. * N * (3. + N) * (3. + N.powu(2))))) / (2. * N.powu(3) * (1. + N).powu(3)) + (12. - 8. / N + 8. / (1. + N)) * S2 + S1 * (40. / N.powu(2) - 8. / (1. + N).powu(2) + 16. * S2 - 16. * Sp2p) + (8. * Sp2p)/(N + N.powu(2)) - 4. * Sp3p - 20. * ZETA3 + ZETA2 * (-32. * S1 + 32. * Sp1p + 32. * (-(1./N) + LN_2));
    let gqq1p_cfnf = (-12. + N * (20. + N * (47. + 3. * N * (2. + N))))/(9. * N.powu(2) * (1. + N).powu(2)) - (40. * S1) / 9. + (8. * S2) / 3.;

    CF * ( CA * gqq1p_cfca + CF * gqq1p_cfcf + 2.0 * TR * (nf as f64) * gqq1p_cfnf)
}

/// Compute the pure-singlet quark-quark anomalous dimension.
///
/// Implements Eq. (3.6) of [\[Vogt:2004mw\]][crate::bib::Vogt:2004mw].
pub fn gamma_ps(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let gqqps1_nfcf = ( -4. * (2. + N * (5. + N)) * (4. + N * (4. + N * (7. + 5. * N))))/((-1. + N) * N.powu(3) * (1. + N).powu(3) * (2. + N).powu(2));
    2.0 * TR * (nf as f64) * CF * gqqps1_nfcf
}

/// Compute the quark-gluon singlet anomalous dimension.
///
/// Implements Eq. (3.7) of [\[Vogt:2004mw\]][crate::bib::Vogt:2004mw].
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp2p = c.get(K::S2h);

    let gqg1_nfca = ( -4. * (16. + N * (64. + N * (104. + N * (128. + N * (85. + N * (36. + N * (25. + N * (15. + N * (6. + N ))))))))))/((-1. + N) * N.powu(3) * (1. + N).powu(3) * (2. + N).powu(3)) - (16. * (3. + 2. * N) * S1) / (2. + 3. * N + N.powu(2)).powu(2) + (4. * (2. + N + N.powu(2)) * S1.powu(2))/(N*(2. + 3. * N + N.powu(2))) - (4. * (2. + N + N.powu(2)) * S2)/(N * (2. + 3. * N + N.powu(2))) + (4. * (2. + N + N.powu(2)) * Sp2p)/(N * (2. + 3. * N + N.powu(2)));
    let gqg1_nfcf = (-2. * (4. + N * (8. + N * (1. + N) * (25. + N * (26. + 5. * N * (2. + N))))))/(N.powu(3) * (1. + N).powu(3) * (2. + N)) + (8. * S1) / N.powu(2) - (4. * (2. + N + N.powu(2)) * S1.powu(2))/(N * (2. + 3. * N + N.powu(2))) + (4. * (2. + N + N.powu(2)) * S2)/(N * (2. + 3. * N + N.powu(2)));
    2.0 * TR * (nf as f64) * (CA * gqg1_nfca + CF * gqg1_nfcf)
}

/// Compute the gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (3.8) of [\[Vogt:2004mw\]][crate::bib::Vogt:2004mw].
pub fn gamma_gq(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp2p = c.get(K::S2h);

    let ggq1_cfcf = (-8. + 2. * N * (-12. + N * (-1. + N * (28. + N * (43. + 6. * N * (5. + 2. * N))))))/((-1. + N) * N.powu(3)*(1. + N).powu( 3)) - (4. * (10. + N * (17. + N * (8. + 5. * N)))*S1) / ((-1. + N) * N * (1. + N).powu(2)) + (4. * (2. + N + N.powu(2)) * S1.powu(2))/(N * (-1. + N.powu(2))) + (4. * (2. + N + N.powu(2)) * S2)/(N*(-1. + N.powu(2)));
    let ggq1_cfca = (-4. * (144. + N * (432. + N * (-152. + N * (-1304. + N * (-1031. + N * (695. + N * (1678. + N * (1400. + N * (621. + 109. * N))))))))))/(9. * N.powu(3) * (1. + N).powu(3) * (-2. + N + N.powu(2)).powu(2)) + (4. * (-12. + N*(-22. + 41. * N + 17. * N.powu(3)))*S1)/(3. * (-1. + N).powu(2) * N.powu(2)*(1. + N)) + ((8. + 4. * N + 4. * N.powu(2)) * S1.powu(2))/(N - N.powu(3)) + ((8. + 4. * N + 4. * N.powu(2))*S2)/(N - N.powu(3)) + (4. * (2. + N + N.powu(2)) * Sp2p)/(N * (-1. + N.powu(2)));
    let ggq1_cfnf = (8. * (16. + N * (27. + N * (13. + 8. * N))))/(9. * (-1. + N) * N * (1. + N).powu(2)) - (8. * (2. + N + N.powu(2)) * S1)/(3. * N * (-1. + N.powu(2)));
    CF * ( CA * ggq1_cfca + CF * ggq1_cfcf + 2.0 * TR * (nf as f64) * ggq1_cfnf)
}

/// Compute the gluon-gluon singlet anomalous dimension.
///
/// Implements Eq. (3.9) of [\[Vogt:2004mw\]][crate::bib::Vogt:2004mw].
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let Sp1p = c.get(K::S1h);
    let Sp2p = c.get(K::S2h);
    let Sp3p = c.get(K::S3h);
    let g3n = c.get(K::G3);
    let ggg1_caca = 16. * g3n - (2. * (576. + N * (1488. + N * (560. + N * (-1248. + N * (-1384. + N * (1663. + N * (4514. + N * (4744. + N * (3030. + N * (1225. + 48. * N * (7. + N))))))))))))/(9. * (-1. + N).powu(2) * N.powu(3) * (1. + N).powu(3) * (2. + N).powu(3)) + S1 * (29.77777777777778 + 16./(-1. + N).powu(2) + 16./(1. + N).powu(2) - 16./(2. + N).powu(2) - 8.*Sp2p) + (16. * (1. + N + N.powu(2)) * Sp2p)/(N * (1. + N)*(-2. + N + N.powu(2))) - 2. * Sp3p - 10. * ZETA3 + ZETA2 * (-16. * S1 + 16. * Sp1p + 16. * (-(1./N) + LN_2));
    let ggg1_canf = (8. * (6. + N * (1. + N) * (28. + N * (1. + N) * (13. + 3. * N * (1. + N)))))/(9. * N.powu(2) * (1. + N).powu(2) * (-2. + N + N.powu(2))) - (40. * S1)/9.;
    let ggg1_cfnf = (2. * (-8. + N * (-8. + N * (-10. + N * (-22. + N * (-3. + N * (6. + N * (8. + N * (4. + N)))))))))/(N.powu(3) * (1. + N).powu(3)*(-2. + N + N.powu(2)));

    CA * CA * ggg1_caca + 2.0 * TR * (nf as f64) * (CA * ggg1_canf + CF * ggg1_cfnf)
    
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gamma_nsp(c, nf) + gamma_ps(c, nf);
    [
        [gamma_qq, gamma_qg(c, nf)],
        [gamma_gq(c, nf), gamma_gg(c, nf)],
    ]
}


#[cfg(test)]
mod tests {
    use crate::cmplx;
    use crate::{anomalous_dimensions::unpolarized::spacelike::as2::*, harmonics::cache::Cache};
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;
    use std::f64::consts::PI;
    use num::traits::Pow;

    const NF: u8 = 5;

    #[test]
    fn gamma_1() {
        // number conservation
        let mut c = Cache::new(cmplx![1., 0.]);
        assert_approx_eq!(f64, gamma_nsm(&mut c, NF).re, 0.0, epsilon=2e-6);

        let mut c = Cache::new(cmplx![2., 0.]);
        let gS1 = gamma_singlet(&mut c, NF);
        // gluon momentum conservation
        assert_approx_eq!(f64, (gS1[0][1] + gS1[1][1]).re, 0.0, epsilon=4e-5);
        // quark momentum conservation
        assert_approx_eq!(f64, (gS1[0][0] + gS1[1][0]).re, 0.0, epsilon=2e-6);
        
        assert_eq!(gS1.len(), 2);
        assert_eq!((gS1[0]).len(), 2);
        assert_eq!((gS1[1]).len(), 2);

        // reference values are obtained from MMa
        // non-singlet sector
        let mut c = Cache::new(cmplx![2., 0.]);
        assert_approx_eq!(
            f64, 
            gamma_nsp(&mut c, NF).re, 
            (-112.0 * CF + 376.0 * CA - 64.0 * (NF as f64)) * CF / 27.0, 
            epsilon=2e-6
        );
        
        // singlet sector
        assert_approx_eq!(
            f64, 
            gamma_ps(&mut c, NF).re, 
            -40.0 * CF * (NF as f64) / 27.0
        );
        // qg
        assert_approx_eq!(
            f64,
            gS1[0][1].re,
            (-74.0 * CF - 35.0 * CA) * (NF as f64) / 27.0
        );
        // gq
        assert_approx_eq!(
            f64,
            gS1[1][0].re,
            (112.0 * CF - 376.0 * CA + 104.0 * (NF as f64)) * CF / 27.0,
            epsilon=1e-13
        );

        // add additional point at (analytical) continuation point
        let check = (
            (34.0 / 27.0 * (-47.0 + 6. * PI.pow(2)) - 16.0 * ZETA3) * CF 
            + (373.0 / 9.0 - 34.0 * PI.pow(2) / 9.0 + 8.0 * ZETA3) * CA 
            - 64.0 * (NF as f64) / 27.0
        )* CF;
        assert_approx_eq!(
            f64, 
            gamma_nsm(&mut c, NF).re, 
            check, 
            epsilon=2e-6);

        let mut c = Cache::new(cmplx![3.,0.]);
        let check = (
            (-34487.0 / 432.0 + 86.0 * PI.pow(2) / 9.0 - 16.0 * ZETA3) * CF
            + (459.0 / 8.0 - 43.0 * PI.pow(2) / 9.0 + 8.0 * ZETA3) * CA
            - 415.0 * (NF as f64) / 108.0
        )* CF;
        assert_approx_eq!(
            f64, 
            gamma_nsp(&mut c, NF).re, 
            check,
            epsilon=2e-6
        );

        assert_approx_eq!(
            f64,
            gamma_ps(&mut c, NF).re,
            -1391.0 * CF * (NF as f64) / 5400.0
        );

        let gS1 = gamma_singlet(&mut c, NF);
        // gq
        assert_approx_eq!(
            f64,
            gS1[1][0].re,
            (973.0 / 432.0 * CF 
            + (2801.0 / 5400.0 - 7.0 * PI.pow(2) / 9.0) * CA
            + 61.0 / 54.0 * (NF as f64)) * CF
        );
        //gg
        assert_approx_eq!(
            f64,
            gS1[1][1].re,
            (-79909.0 / 3375.0 + 194.0 * PI.pow(2) / 45.0 - 8.0 * ZETA3)
            * CA.pow(2)
            - 967.0 / 270.0 * CA * (NF as f64)
            + 541.0 / 216.0 * CF * (NF as f64),
            epsilon=1e-4  // lower numerical precision than python code?
        );
        
        let mut c = Cache::new(cmplx![4.,0.]);
        let gS1 = gamma_singlet(&mut c, NF);
        assert_approx_eq!(
            f64,
            gS1[0][1].re,
            (-56317.0 / 18000.0 * CF + 16387.0 / 9000.0 * CA) * (NF as f64),
            epsilon=1e-14
        )
    }   
}