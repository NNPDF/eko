use num::complex::Complex;

use crate::constants::CF;
use crate::harmonics::cache::{Cache, K};

/// Compute the leading-order non-singlet anomalous dimension.
///
/// Implements Eq. (3.4) of :cite:`Moch:2004pa`.
pub fn gamma_ns(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let gamma = -(3.0 - 4.0 * S1 + 2.0 / N / (N + 1.0));
    CF * gamma
}
