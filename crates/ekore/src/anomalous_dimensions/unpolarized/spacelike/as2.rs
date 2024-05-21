//! NLO QCD

use::num::complex::Complex;
use std::f64::consts::LN_2;

use crate::constants::{CA, CF, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};


/// Compute the valence-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsm(c: &mut Cache, _nf: u8) -> Complex<f64> {
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
    
    CF * ( CA * gqq1m_cfca + CF * gqq1m_cfcf + 2.0 * TR * (_nf as f64) * gqq1m_cfnf )
}


/// Compute the singlet-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsp(c: &mut Cache, _nf: u8) -> Complex<f64> {
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

    CF * ( CA * gqq1p_cfca + CF * gqq1p_cfcf + 2.0 * TR * (_nf as f64) * gqq1p_cfnf)


}

#[cfg(test)]
mod tests {
    use crate::cmplx;
    use crate::constants::{CA, CF, TR, ZETA2, ZETA3};
    use crate::{anomalous_dimensions::unpolarized::spacelike::as2::*, harmonics::cache::Cache};
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;
    use std::f64::consts::{PI, LN_2};
    use num::traits::Pow;

    const NF: u8 = 5;

    #[test]
    fn try_test() {
        let mut c = Cache::new(cmplx![1., 0.]);
        let me = gamma_nsm(&mut c, NF);
        assert_approx_eq!(f64, me.re, 0.0, epsilon=2e-6);

        let mut c = Cache::new(cmplx![2., 0.]);
        let me = gamma_nsp(&mut c, NF);
        let check = (-112.0 * CF + 376.0 * CA - 64.0 * (NF as f64)) * CF / 27.0;
        assert_approx_eq!(f64, me.re, check, epsilon=2e-6);
    
    }

    
}