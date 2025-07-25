//! Harmonic sums of weight 3.
use num::complex::Complex;
use num::traits::Pow;
use std::f64::consts::LN_2;

use crate::constants::{ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::g_functions::g3;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_3(N)$.
///
/// $$S_3(N) = \sum\limits_{j=1}^N \frac 1 {j^3} = \frac 1 2 \psi_2(N+1)+\zeta(3)$$
/// with $\psi_2(N)$ the 2nd polygamma function and $\zeta$ the Riemann zeta function.
pub fn S3(N: Complex<f64>) -> Complex<f64> {
    0.5 * cern_polygamma(N + 1.0, 2) + ZETA3
}

/// Analytic continuation of harmonic sum $S_{-3}(N)$ for even moments.
pub fn Sm3e(c: &mut Cache) -> Complex<f64> {
    let hS3 = c.get(K::S3);
    let hS3h = c.get(K::S3h);
    1. / (2.).pow(2) * hS3h - hS3
}

/// Analytic continuation of harmonic sum $S_{-3}(N)$ for odd moments.
pub fn Sm3o(c: &mut Cache) -> Complex<f64> {
    let hS3 = c.get(K::S3);
    let hS3mh = c.get(K::S3mh);
    1. / (2.).pow(2) * hS3mh - hS3
}

/// Analytic continuation of harmonic sum $S_{-2,1}(N)$ for even moments.
pub fn Sm21e(c: &mut Cache) -> Complex<f64> {
    let hSm1 = c.get(K::Sm1e);
    let mut cp1 = Cache::new(c.n() + 1.);
    ZETA2 * hSm1 - 5. / 8. * ZETA3 + ZETA2 * LN_2 - g3(&mut cp1)
}

/// Analytic continuation of harmonic sum $S_{-2,1}(N)$ for odd moments.
pub fn Sm21o(c: &mut Cache) -> Complex<f64> {
    let hSm1 = c.get(K::Sm1o);
    let mut cp1 = Cache::new(c.n() + 1.);
    ZETA2 * hSm1 - 5. / 8. * ZETA3 + ZETA2 * LN_2 + g3(&mut cp1)
}
