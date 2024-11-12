//! |NLO| |QCD|.

use num::complex::Complex;

use super::super::super::unpolarized::spacelike::as2::gamma_nsm as unpol_nsm;
use super::super::super::unpolarized::spacelike::as2::gamma_nsp as unpol_nsp;
use crate::constants::{CA, CF, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};

pub fn gamma_nsm(c: &mut Cache, nf: u8) -> Complex<f64> {
    unpol_nsp(c, nf)
}

pub fn gamma_nsp(c: &mut Cache, nf: u8) -> Complex<f64> {
    unpol_nsm(c, nf)
}

/// Compute the pure-singlet quark-quark anomalous dimension.
///
/// Implements Eq. (A.3) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_ps(c: &mut Cache, nf: u8) -> Complex<f64> {
    let n = c.n();
    let gqqps1_nfcf = (2. * (n + 2.) * (1. + 2. * n + n.powu(3))) / ((1. + n).powu(3) * n.powu(3));
    4.0 * TR * (nf as f64) * CF * gqqps1_nfcf
}

/// Compute the quark-gluon singlet anomalous dimension.
///
/// Implements Eq. (A.4) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp2m = c.get(K::S2mh);

    #[rustfmt::skip]
    let gqg1_nfca = (
        (S1.powu(2) - S2 + Sp2m) * (n - 1.) / (n * (n + 1.))
        - 4. * S1 / (n * (1. + n).powu(2))
        - (-2. - 7. * n + 3. * n.powu(2) - 4. * n.powu(3) + n.powu(4) + n.powu(5)) / (n.powu(3) * (1. + n).powu(3))
    ) * 2.0;
    #[rustfmt::skip]
    let gqg1_nfcf = (
        (-(S1.powu(2)) + S2 + 2. * S1 / n) * (n - 1.) / (n * (n + 1.))
        - (n - 1.)
        * (1. + 3.5 * n + 4. * n.powu(2) + 5. * n.powu(3) + 2.5 * n.powu(4))
        / (n.powu(3) * (1. + n).powu(3))
        + 4. * (n - 1.) / (n.powu(2) * (1. + n).powu(2))
    ) * 2.;
    4.0 * TR * (nf as f64) * (CA * gqg1_nfca + CF * gqg1_nfcf)
}

/// Compute the gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (A.5) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_gq(c: &mut Cache, nf: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sp2m = c.get(K::S2mh);
    #[rustfmt::skip]
    let ggq1_cfcf = (
        (2. * (S1.powu(2) + S2) * (n + 2.)) / (n * (n + 1.))
        - (2. * S1 * (n + 2.) * (1. + 3. * n)) / (n * (1. + n).powu(2))
        - ((n + 2.) * (2. + 15. * n + 8. * n.powu(2) - 12.0 * n.powu(3) - 9.0 * n.powu(4)))
        / (n.powu(3) * (1. + n).powu(3))
        + 8. * (n + 2.) / (n.powu(2) * (1. + n).powu(2))
    ) * 0.5;
    #[rustfmt::skip]
    let ggq1_cfca = -(
        -(-(S1.powu(2)) - S2 + Sp2m) * (n + 2.) / (n * (n + 1.))
        - S1 * (12. + 22. * n + 11. * n.powu(2)) / (3. * n.powu(2) * (n + 1.))
        + (36. + 72. * n + 41. * n.powu(2) + 254. * n.powu(3) + 271. * n.powu(4) + 76. * n.powu(5))
            / (9. * n.powu(3) * (1. + n).powu(3))
    );
    #[rustfmt::skip]
    let ggq1_cfnf =(-S1 * (n + 2.)) / (3. * n * (n + 1.)) + ((n + 2.) * (2. + 5. * n)) / (
        9. * n * (1. + n).powu(2)
    );
    4. * CF * (CA * ggq1_cfca + CF * ggq1_cfcf + 4.0 * TR * (nf as f64) * ggq1_cfnf)
}

/// Compute the gluon-gluon singlet anomalous dimension.
///
/// Implements Eq. (A.6) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let Sp1m = c.get(K::S1mh);
    let Sp2m = c.get(K::S2mh);
    let Sp3m = c.get(K::S3mh);
    let S1h = c.get(K::S1h);
    let g3 = c.get(K::G3);
    let SSCHLM = ZETA2 / 2. * (Sp1m - S1h + 2. / n) - S1 / n.powu(2) - g3 - 5. * ZETA3 / 8.;
    #[rustfmt::skip]
    let ggg1_caca = (
        -4. * S1 * Sp2m
        - Sp3m
        + 8. * SSCHLM
        + 8. * Sp2m / (n * (n + 1.))
        + 2.0
        * S1
        * (72. + 144. * n + 67. * n.powu(2) + 134. * n.powu(3) + 67. * n.powu(4))
        / (9. * n.powu(2) * (n + 1.).powu(2))
        - (144. + 258. * n + 7. * n.powu(2) + 698. * n.powu(3) + 469. * n.powu(4) + 144. * n.powu(5) + 48. * n.powu(6))
        / (9. * n.powu(3) * (1. + n).powu(3))
    ) * 0.5;
    #[rustfmt::skip]
    let ggg1_canf = (
        -5. * S1 / 9.
        + (-3. + 13. * n + 16. * n.powu(2) + 6.* n.powu(3) + 3. * n.powu(4)) / (9. * n.powu(2) * (1. + n).powu(2))
    ) * 4.;
    #[rustfmt::skip]
    let ggg1_cfnf = (4. + 2. * n - 8. * n.powu(2) + n.powu(3) + 5. * n.powu(4) + 3. * n.powu(5) + n.powu(6)) / (
        n.powu(3) * (1. + n).powu(3)
    );
    4. * (CA * CA * ggg1_caca + TR * (nf as f64) * (CA * ggg1_canf + CF * ggg1_cfnf))
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
    use super::super::as1::gamma_gq as as1_gamma_gq;
    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::traits::Pow;
    use num::Zero;
    use std::f64::consts::PI;

    const NF: u8 = 5;

    #[test]
    fn physical_constraints() {
        // qg_helicity_conservation
        let mut c = Cache::new(cmplx!(1., 0.));
        assert_approx_eq_cmplx!(f64, gamma_qg(&mut c, NF), Complex::zero(), epsilon = 2e-6);

        // qg momentum
        let mut c = Cache::new(cmplx!(1., 0.));
        let gS1 = gamma_singlet(&mut c, NF);
        assert_approx_eq_cmplx!(
            f64,
            gS1[0][0],
            cmplx!(12. * TR * (NF as f64) * CF, 0.),
            epsilon = 2e-6
        );
    }

    #[test]
    fn N2() {
        let mut c = Cache::new(cmplx!(2., 0.));

        // singlet sector
        let gS1 = gamma_singlet(&mut c, NF);
        // ps
        assert_approx_eq_cmplx!(
            f64,
            -gamma_ps(&mut c, NF),
            cmplx!(-4.0 * CF * TR * (NF as f64) * 13. / 27.0, 0.)
        );
        // qg
        assert_approx_eq_cmplx!(
            f64,
            -gS1[0][1],
            cmplx!(
                4. * (NF as f64)
                    * (0.574074074 * CF - 2. * CA * (-7. / 18. + 1. / 6. * (5. - PI.pow(2) / 3.)))
                    * TR,
                0.
            ),
            epsilon = 1e-9
        );
        // gq
        assert_approx_eq_cmplx!(
            f64,
            -gS1[1][0],
            cmplx!(
                4. * (-2.074074074074074 * CF.pow(2)
                    + CA * CF * (29. / 54. - 2. / 3. * (1. / 2. - PI.pow(2) / 3.))
                    + (4. * CF * (NF as f64) * TR) / 27.),
                0.
            ),
            epsilon = 1e-13
        );
    }

    #[test]
    fn axial_anomaly() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let me_ps = gamma_ps(&mut c, NF);
        let as1_gq = -2.0 * (NF as f64) * as1_gamma_gq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me_ps, as1_gq);
        // TODO: activate this test once the beta function will be available
        // let me_gg = gamma_gg(&mut c, NF);
        // let beta = -1.0 * beta_qcd_as2(NF);
        // assert_approx_eq_cmplx!(f64, me_gg, beta, rel=9e-7);
    }
}
