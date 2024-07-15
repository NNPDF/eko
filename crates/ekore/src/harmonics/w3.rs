//! Harmonic sums of weight 3.
use num::complex::Complex;

use crate::constants::ZETA3;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_3(N)$.
///
/// $$S_3(N) = \sum\limits_{j=1}^N \frac 1 {j^3} = \frac 1 2 \psi_2(N+1)+\zeta(3)$$
/// with $\psi_2(N)$ the 2nd polygamma function and $\zeta$ the Riemann zeta function.
pub fn S3(N: Complex<f64>) -> Complex<f64> {
    0.5 * cern_polygamma(N + 1.0, 2) + ZETA3
}
