//! Harmonic sums of weight 4.
use num::complex::Complex;

use crate::constants::ZETA4;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_4(N)$.
///
/// $$S_4(N) = \sum\limits_{j=1}^N \frac 1 {j^4} = - \frac 1 6 \psi_3(N+1)+\zeta(4)$$
/// with $\psi_3(N)$ the 3rd polygamma function and $\zeta$ the Riemann zeta function.
pub fn S4(N: Complex<f64>) -> Complex<f64> {
    ZETA4 - 1.0 / 6.0 * cern_polygamma(N + 1.0, 3)
}
