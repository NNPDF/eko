use num::complex::Complex;

use crate::constants::{CA, CF, TR};
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

/// Compute the leading-order quark-gluon anomalous dimension.
///
/// Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let gamma = -(N.powu(2) + N + 2.0) / (N * (N + 1.0) * (N + 2.0));
    2.0 * TR * 2.0 * (nf as f64) * gamma
}

/// Compute the leading-order gluon-quark anomalous dimension.
///
/// Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
pub fn gamma_gq(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n;
    let gamma = -(N.powu(2) + N + 2.0) / (N * (N + 1.0) * (N - 1.0));
    2.0 * CF * gamma
}

/// Compute the leading-order gluon-gluon anomalous dimension.
///
/// Implements Eq. (3.5) of :cite:`Vogt:2004mw`.
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let gamma = S1 - 1.0 / N / (N - 1.0) - 1.0 / (N + 1.0) / (N + 2.0);
    CA * (4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * TR * (nf as f64)
}

/// Compute the leading-order singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gamma_ns(c, nf);
    [
        [gamma_qq, gamma_qg(c, nf)],
        [gamma_gq(c, nf), gamma_gg(c, nf)],
    ]
}
