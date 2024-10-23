//! The $O(a_s^1a_{em}^1)$ Altarelli-Parisi splitting kernels.
use num::complex::Complex;

use crate::constants::{CA, CF};
use crate::harmonics::cache::{Cache, K};

/// Compute the $O(a_s^1a_{em}^1)$ photon-quark anomalous dimension.
///
/// Implements Eq. (36) of
pub fn gamma_phq(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);

    #[rustfmt::skip]
    let tmp_const =
        2.0
        * (
            -4.0
            - 12.0 * N
            - N.powu(2)
            + 28.0 * N.powu(3)
            + 43.0 * N.powu(4)
            + 30.0 * N.powu(5)
            + 12.0 * N.powu(6)
        ) / ((-1.0 + N) * N.powu(3) * (1.0 + N).powu(3));

    #[rustfmt::skip]
    let tmp_S1 = -4.0
        * (10.0 + 27.0 * N + 25.0 * N.powu(2) + 13.0 * N.powu(3) + 5.0 * N.powu(4))
        / ((-1.0 + N) * N * (1.0 + N).powu(3));

    let tmp_S12 = 4.0 * (2.0 + N + N.powu(2)) / ((-1.0 + N) * N * (1.0 + N));
    let tmp_S2 = 4.0 * (2.0 + N + N.powu(2)) / ((-1.0 + N) * N * (1.0 + N));

    CF * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1.powu(2) + tmp_S2 * S2)
}

/// Compute the $O(a_s^1a_{em}^1)$ quark-photon anomalous dimension.
///
/// Implements Eq. (26) of
pub fn gamma_qph(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);

    let tmp_const = -2.0
        * (4.0
            + 8.0 * N
            + 25.0 * N.powu(2)
            + 51.0 * N.powu(3)
            + 36.0 * N.powu(4)
            + 15.0 * N.powu(5)
            + 5.0 * N.powu(6))
        / (N.powu(3) * (1.0 + N).powu(3) * (2.0 + N));

    let tmp_S1 = 8.0 / N.powu(2);
    let tmp_S12 = -4.0 * (2.0 + N + N.powu(2)) / (N * (1.0 + N) * (2.0 + N));
    let tmp_S2 = 4.0 * (2.0 + N + N.powu(2)) / (N * (1.0 + N) * (2.0 + N));

    2.0 * (nf as f64) * CA * CF * (tmp_const + tmp_S1 * S1 + tmp_S12 * S1.powu(2) + tmp_S2 * S2)
}
