use num::complex::Complex;

use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum $S_1(N)$.
///
/// $$S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E$$
/// with $\psi_0(N)$ the digamma function and $\gamma_E$ the Euler-Mascheroni constant.
pub fn S1(N: Complex<f64>) -> Complex<f64> {
    cern_polygamma(N + 1.0, 0) + 0.577_215_664_901_532_9
}
