//! Harmonic sums of weight 2.
use num::complex::Complex;

use crate::constants::ZETA2;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_2(N)$.
///
/// $$S_2(N) = \sum\limits_{j=1}^N \frac 1 {j^2} = -\psi_1(N+1)+\zeta(2)$$
/// with $\psi_1(N)$ the trigamma function and $\zeta$ the Riemann zeta function.
pub fn S2(N: Complex<f64>) -> Complex<f64> {
    -cern_polygamma(N + 1.0, 1) + ZETA2
}
