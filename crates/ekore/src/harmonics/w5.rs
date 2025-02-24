//! Harmonic sums of weight 5.
use num::complex::Complex;

use crate::constants::ZETA5;
use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_5(N)$.
///
/// $$S_5(N) = \sum\limits_{j=1}^N \frac 1 {j^5} = - \frac 1 24 \psi_4(N+1)+\zeta(5)$$
/// with $\psi_4(N)$ the 4rd polygamma function and $\zeta$ the Riemann zeta function.
pub fn S5(N: Complex<f64>) -> Complex<f64> {
    ZETA5 + 1.0 / 24.0 * cern_polygamma(N + 1.0, 4)
}
