//! |LO| |QCD|.

use num::complex::Complex;

use crate::constants::{CA, CF, TR};
use crate::harmonics::cache::{Cache, K};

/// Compute the non-singlet anomalous dimension.
///
/// Implements Eq. (3.4) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_ns(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let gamma = -(3.0 - 4.0 * S1 + 2.0 / N / (N + 1.0));
    CF * gamma
}

/// Compute the quark-gluon anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Vogt:2004mw\]](crate::bib::Vogt2004mw).
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let gamma = -(N.powu(2) + N + 2.0) / (N * (N + 1.0) * (N + 2.0));
    2.0 * TR * 2.0 * (nf as f64) * gamma
}

/// Compute the gluon-quark anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Vogt:2004mw\]](crate::bib::Vogt2004mw).
pub fn gamma_gq(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let gamma = -(N.powu(2) + N + 2.0) / (N * (N + 1.0) * (N - 1.0));
    2.0 * CF * gamma
}

/// Compute the gluon-gluon anomalous dimension.
///
/// Implements Eq. (3.5) of [\[Vogt:2004mw\]](crate::bib::Vogt2004mw).
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let gamma = S1 - 1.0 / N / (N - 1.0) - 1.0 / (N + 1.0) / (N + 2.0);
    CA * (4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * TR * (nf as f64)
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gamma_ns(c, nf);
    [
        [gamma_qq, gamma_qg(c, nf)],
        [gamma_gq(c, nf), gamma_gg(c, nf)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;
    const NF: u8 = 5;

    #[test]
    fn number_conservation() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let me = gamma_ns(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn quark_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let me = gamma_ns(&mut c, NF) + gamma_gq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn gluon_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let me = gamma_qg(&mut c, NF) + gamma_gg(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn gamma_qg_() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let me = gamma_qg(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, cmplx!(-20. / 3., 0.), ulps = 32, epsilon = 1e-12);
    }

    #[test]
    fn gamma_gq_() {
        const N: Complex<f64> = cmplx!(0., 1.);
        let mut c = Cache::new(N);
        let me = gamma_gq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, cmplx!(4. / 3.0, -4. / 3.0), ulps = 32);
    }

    #[test]
    fn gamma_gg_() {
        const N: Complex<f64> = cmplx!(0., 1.);
        let mut c = Cache::new(N);
        let me = gamma_gg(&mut c, NF);
        assert_approx_eq_cmplx!(
            f64,
            me,
            cmplx!(5.195725159621, 10.52008856962),
            ulps = 32,
            epsilon = 1e-11
        );
    }
}
