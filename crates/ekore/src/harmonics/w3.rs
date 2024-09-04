//! Harmonic sums of weight 3.
use num::complex::Complex;
use std::f64::consts::LN_2;

use crate::constants::{ZETA2, ZETA3};
use crate::harmonics::g_functions::g3;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_3(N)$.
///
/// $$S_3(N) = \sum\limits_{j=1}^N \frac 1 {j^3} = \frac 1 2 \psi_2(N+1)+\zeta(3)$$
/// with $\psi_2(N)$ the 2nd polygamma function and $\zeta$ the Riemann zeta function.
pub fn S3(N: Complex<f64>) -> Complex<f64> {
    0.5 * cern_polygamma(N + 1.0, 2) + ZETA3
}

/// Analytic continuation of harmonic sum $S_{-2,1}(N)$ for even moments.
pub fn Sm21e(N: Complex<f64>, hS1: Complex<f64>, hSm1: Complex<f64>) -> Complex<f64> {
    let eta = 1.;
    -eta * g3(N + 1., hS1 + 1. / (N + 1.)) + ZETA2 * hSm1 - 5. / 8. * ZETA3 + ZETA2 * LN_2
}

/// Analytic continuation of harmonic sum $S_{-2,1}(N)$ for odd moments.
pub fn Sm21o(N: Complex<f64>, hS1: Complex<f64>, hSm1: Complex<f64>) -> Complex<f64> {
    let eta = -1.;
    -eta * g3(N + 1., hS1 + 1. / (N + 1.)) + ZETA2 * hSm1 - 5. / 8. * ZETA3 + ZETA2 * LN_2
}
