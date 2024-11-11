//! The $O(a_s^1a_{em}^1)$ Altarelli-Parisi splitting kernels.
use crate::cmplx;
use num::complex::Complex;

use crate::constants::{
    ed2, eu2, uplike_flavors, ChargeCombinations, CA, CF, NC, TR, ZETA2, ZETA3,
};
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

    #[rustfmt::skip]
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

/// Compute the $O(a_s^1a_{em}^1)$ singlet-like non singlet anomalous dimension.
///
/// Implements Eqs. (33-34) of
pub fn gamma_nsp(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S1h = c.get(K::S1h);
    let S2h = c.get(K::S2h);
    let S3h = c.get(K::S3h);
    let S1p1h = c.get(K::S1ph);
    let S2p1h = c.get(K::S2ph);
    let S3p1h = c.get(K::S3ph);

    let g3N = c.get(K::G3);
    let g3Np2 = c.get(K::G3p2);

    #[rustfmt::skip]
    let result = 32.0 * ZETA2 * S1h - 32.0 * ZETA2 * S1p1h
        + 8.0 / (N + N.powu(2)) * S2h
        - 4.0 * S3h + (24.0 + 16.0 / (N + N.powu(2))) * S2
        - 32.0 * S3 - 8.0 / (N + N.powu(2)) * S2p1h
        + S1 * (16.0 * (3.0 / N.powu(2) - 3.0 / (1.0 + N).powu(2) + 2.0 * ZETA2) - 16.0 * S2h
            - 32.0 * S2 + 16.0 * S2p1h )
        + (-8.0 + N * (-32.0 + N * ( -8.0 - 3.0 * N * (3.0 + N) * (3.0 + N.powu(2)) - 48.0 * (1.0 + N).powu(2) * ZETA2)))
        / (N.powu(3) * (1.0 + N).powu(3))
        + 32.0 * (g3N + g3Np2) + 4.0 * S3p1h - 16.0 * ZETA3;

    CF * result
}

/// Compute the $O(a_s^1a_{em}^1)$ valence-like non singlet anomalous dimension.
///
/// Implements Eqs. (33-34) of
pub fn gamma_nsm(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S1h = c.get(K::S1h);
    let S2h = c.get(K::S2h);
    let S3h = c.get(K::S3h);
    let S1p1h = c.get(K::S1ph);
    let S2p1h = c.get(K::S2ph);
    let S3p1h = c.get(K::S3ph);
    let g3N = c.get(K::G3);
    let g3Np2 = c.get(K::G3p2);

    #[rustfmt::skip]
    let result =
        -32.0 * ZETA2 * S1h
        - 8.0 / (N + N.powu(2)) * S2h
        + (24.0 + 16.0 / (N + N.powu(2))) * S2
        + 8.0 / (N + N.powu(2)) * S2p1h
        + S1
        * (
            16.0 * (-1.0 / N.powu(2) + 1.0 / (1.0 + N).powu(2) + 2.0 * ZETA2)
            + 16.0 * S2h
            - 32.0 * S2
            - 16.0 * S2p1h
        )
        + (
            72.0
            + N
            * (
                96.0
                - 3.0 * N * (8.0 + 3.0 * N * (3.0 + N) * (3.0 + N.powu(2)))
                + 48.0 * N * (1.0 + N).powu(2) * ZETA2
            )
        )
        / (3.0 * N.powu(3) * (1.0 + N).powu(3))
        - 32.0 * (g3N + g3Np2)
        + 32.0 * ZETA2 * S1p1h
        + 4.0 * S3h
        - 32.0 * S3
        - 4.0 * S3p1h
        - 16.0 * ZETA3;

    CF * result
}

/// Compute the $O(a_s^1a_{em}^1)$ singlet sector.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 4]; 4] {
    let cc = ChargeCombinations { nf };
    // let e2avg = cc.e2avg();
    // let vue2m = cc.vue2m();
    // let vde2m = cc.vde2m();
    // let e2delta = cc.e2delta();
    let e2_tot = nf as f64 * cc.e2avg();

    [
        [
            e2_tot * gamma_gg(c, nf),
            e2_tot * gamma_gph(c, nf),
            cc.e2avg() * gamma_gq(c, nf),
            cc.vue2m() * gamma_gq(c, nf),
        ],
        [
            e2_tot * gamma_phg(c, nf),
            gamma_phph(c, nf),
            cc.e2avg() * gamma_phq(c, nf),
            cc.vue2m() * gamma_phq(c, nf),
        ],
        [
            cc.e2avg() * gamma_qg(c, nf),
            cc.e2avg() * gamma_qph(c, nf),
            cc.e2avg() * gamma_nsp(c, nf),
            cc.vue2m() * gamma_nsp(c, nf),
        ],
        [
            cc.vde2m() * gamma_qg(c, nf),
            cc.vde2m() * gamma_qph(c, nf),
            cc.vde2m() * gamma_nsp(c, nf),
            cc.e2delta() * gamma_nsp(c, nf),
        ],
    ]
}

/// Compute the $O(a_s^1a_{em}^1)$ valence sector.
pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let cc = ChargeCombinations { nf };
    [
        [cc.e2avg() * gamma_nsm(c, nf), cc.vue2m() * gamma_nsm(c, nf)],
        [
            cc.vde2m() * gamma_nsm(c, nf) * gamma_nsm(c, nf),
            cc.e2delta() * gamma_nsm(c, nf),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;
    const NF: u8 = 5;

    #[test]
    fn number_conservation() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let me = gamma_nsm(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-4);
    }

    #[test]
    fn gluon_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);

        for nf in 2..7 {
            let nu = uplike_flavors(nf);
            let nd = nf - nu;
            assert_approx_eq_cmplx!(
                f64,
                eu2 * gamma_qg(&mut c, nu)
                    + ed2 * gamma_qg(&mut c, nd)
                    + (nu as f64 * eu2 + nd as f64 * ed2) * gamma_phg(&mut c, nf)
                    + (nu as f64 * eu2 + nd as f64 * ed2) * gamma_gg(&mut c, nf),
                cmplx!(0., 0.),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn photon_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);

        for nf in 2..7 {
            let nu = uplike_flavors(nf);
            let nd = nf - nu;
            assert_approx_eq_cmplx!(
                f64,
                eu2 * gamma_qph(&mut c, nu)
                    + ed2 * gamma_qph(&mut c, nd)
                    + gamma_phph(&mut c, nf)
                    + (nu as f64 * eu2 + nd as f64 * ed2) * gamma_gph(&mut c, nf),
                cmplx!(0., 0.),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn quark_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let me = gamma_nsp(&mut c, NF) + gamma_gq(&mut c, NF) + gamma_phq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-4);
    }
}
