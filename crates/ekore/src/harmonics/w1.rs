use num::complex::Complex;

use crate::harmonics::polygamma::cern_polygamma;

/// Compute the harmonic sum :math:`S_1(N)`.
///
/// .. math::
///     S_1(N) = \sum\limits_{j=1}^N \frac 1 j = \psi_0(N+1)+\gamma_E
/// with :math:`\psi_0(N)` the digamma function and :math:`\gamma_E` the Euler-Mascheroni constant.
pub fn S1(N: Complex<f64>) -> Complex<f64> {
    cern_polygamma(N + 1.0, 0) + 0.5772156649015328606065120
}
