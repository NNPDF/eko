//! Harmonic sums of weight 2.
use num::complex::Complex;

use crate::constants::ZETA2;
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_2(N)$.
///
/// $$S_2(N) = \sum\limits_{j=1}^N \frac 1 {j^2} = -\psi_1(N+1)+\zeta(2)$$
/// with $\psi_1(N)$ the trigamma function and $\zeta$ the Riemann zeta function.
pub fn S2(N: Complex<f64>) -> Complex<f64> {
    -cern_polygamma(N + 1.0, 1) + ZETA2
}

/// Analytic continuation of harmonic sum $S_{-2}(N)$ for even moments.
pub fn Sm2e(c: &mut Cache) -> Complex<f64> {
    let hS2 = c.get(K::S2);
    let hS2h = c.get(K::S2h);
    1. / 2. * hS2h - hS2
}

/// Analytic continuation of harmonic sum $S_{-2}(N)$ for odd moments.
pub fn Sm2o(c: &mut Cache) -> Complex<f64> {
    let hS2 = c.get(K::S2);
    let hS2mh = c.get(K::S2mh);
    1. / 2. * hS2mh - hS2
}
