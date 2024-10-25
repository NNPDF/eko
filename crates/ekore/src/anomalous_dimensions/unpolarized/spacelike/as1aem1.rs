//! The $O(a_s^1a_{em}^1)$ Altarelli-Parisi splitting kernels.
use crate::cmplx;
use num::complex::Complex;

use crate::constants::{ed2, eu2, uplike_flavors, CA, CF, NC, TR};
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

/// Compute the $O(a_s^1a_{em}^1)$ gluon-photon anomalous dimension.
///
/// Implements Eq. (27) of
pub fn gamma_gph(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    CF * CA
        * (8.0 * (-4.0 + N * (-4.0 + N * (-5.0 + N * (-10.0 + N + 2.0 * N.powu(2) * (2.0 + N))))))
        / (N.powu(3) * (1.0 + N).powu(3) * (-2.0 + N + N.powu(2)))
}

/// Compute the $O(a_s^1a_{em}^1)$ photon-gluon anomalous dimension.
///
/// Implements Eq. (30) of
pub fn gamma_phg(c: &mut Cache, nf: u8) -> Complex<f64> {
    TR / CF / CA * (NC as f64) * gamma_gph(c, nf)
}

/// Compute the $O(a_s^1a_{em}^1)$ quark-gluon singlet anomalous dimension.
///
/// Implements Eq. (29) of
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    TR / CF / CA * (NC as f64) * gamma_qph(c, nf)
}

/// Compute the $O(a_s^1a_{em}^1)$ gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (35) of
pub fn gamma_gq(c: &mut Cache, nf: u8) -> Complex<f64> {
    gamma_phq(c, nf)
}

/// Compute the $O(a_s^1a_{em}^1)$ photon-photon singlet anomalous dimension.
///
/// Implements Eq. (28) of
pub fn gamma_phph(_c: &mut Cache, nf: u8) -> Complex<f64> {
    let nu = uplike_flavors(nf);
    let nd = nf - nu;
    cmplx!(4.0 * CF * CA * ((nu as f64) * eu2 + (nd as f64) * ed2), 0.)
}

/// Compute the $O(a_s^1a_{em}^1)$ gluon-gluon singlet anomalous dimension.
///
/// Implements Eq. (31) of
pub fn gamma_gg(_c: &mut Cache, _nf: u8) -> Complex<f64> {
    cmplx!(4.0 * TR * (NC as f64), 0.)
}
