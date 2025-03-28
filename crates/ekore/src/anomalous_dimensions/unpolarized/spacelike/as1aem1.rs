//! |LO| |QED| x |LO| |QCD|.
use crate::cmplx;
use num::complex::Complex;

use crate::constants::{ChargeCombinations, CA, CF, ED2, EU2, NC, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{recursive_harmonic_sum, Cache, K};
use std::f64::consts::PI;

/// Compute the photon-quark anomalous dimension.
///
/// Implements Eq. (36) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
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

/// Compute the quark-photon anomalous dimension.
///
/// Implements Eq. (26) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
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

/// Compute the gluon-photon anomalous dimension.
///
/// Implements Eq. (27) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_gph(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    CF * CA
        * (8.0 * (-4.0 + N * (-4.0 + N * (-5.0 + N * (-10.0 + N + 2.0 * N.powu(2) * (2.0 + N))))))
        / (N.powu(3) * (1.0 + N).powu(3) * (-2.0 + N + N.powu(2)))
}

/// Compute the photon-gluon anomalous dimension.
///
/// Implements Eq. (30) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_phg(c: &mut Cache, nf: u8) -> Complex<f64> {
    TR / CF / CA * (NC as f64) * gamma_gph(c, nf)
}

/// Compute the quark-gluon singlet anomalous dimension.
///
/// Implements Eq. (29) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    TR / CF / CA * (NC as f64) * gamma_qph(c, nf)
}

/// Compute the gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (35) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_gq(c: &mut Cache, nf: u8) -> Complex<f64> {
    gamma_phq(c, nf)
}

/// Compute the photon-photon singlet anomalous dimension.
///
/// Implements Eq. (28) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_phph(_c: &mut Cache, nf: u8) -> Complex<f64> {
    let cc = ChargeCombinations { nf };
    cmplx!(
        4.0 * CF * CA * ((cc.nu() as f64) * EU2 + (cc.nd() as f64) * ED2),
        0.
    )
}

/// Compute the gluon-gluon singlet anomalous dimension.
///
/// Implements Eq. (31) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_gg(_c: &mut Cache, _nf: u8) -> Complex<f64> {
    cmplx!(4.0 * TR * (NC as f64), 0.)
}

/// Shift for $g_3(N)$ by 2.
///
/// This is $g_3(N+2) - g_3(N)$.
fn g3_shift(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    (6. * (n + 1.) * (2. * n + 1.) * S1 + n * (-PI.powi(2) * (n + 1.).powu(2) - 6. * n))
        / (6. * n.powu(2) * (n + 1.).powu(3))
}

/// Compute the singlet-like non-singlet anomalous dimension.
///
/// Implements Eqs. (33-34) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_nsp(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S1h = c.get(K::S1h);
    let S2h = c.get(K::S2h);
    let S3h = c.get(K::S3h);
    let S1p1h = recursive_harmonic_sum(c.get(K::S1mh), (N - 1.) / 2., 1, 1);
    let S2p1h = recursive_harmonic_sum(c.get(K::S2mh), (N - 1.) / 2., 1, 2);
    let S3p1h = recursive_harmonic_sum(c.get(K::S3mh), (N - 1.) / 2., 1, 3);

    let g3N = c.get(K::G3);
    let g3Np2 = g3N + g3_shift(c);

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

/// Compute the valence-like non-singlet anomalous dimension.
///
/// Implements Eqs. (33-34) of [\[deFlorian:2015ujt\]][crate::bib::deFlorian2015ujt].
pub fn gamma_nsm(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S1h = c.get(K::S1h);
    let S2h = c.get(K::S2h);
    let S3h = c.get(K::S3h);
    let S1p1h = recursive_harmonic_sum(c.get(K::S1mh), (N - 1.) / 2., 1, 1);
    let S2p1h = recursive_harmonic_sum(c.get(K::S2mh), (N - 1.) / 2., 1, 2);
    let S3p1h = recursive_harmonic_sum(c.get(K::S3mh), (N - 1.) / 2., 1, 3);
    let g3N = c.get(K::G3);
    let g3Np2 = g3N + g3_shift(c);

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

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 4]; 4] {
    let cc = ChargeCombinations { nf };
    let e2_tot = nf as f64 * cc.e2avg();
    let gq = gamma_gq(c, nf);
    let phq = gamma_phq(c, nf);
    let qg = gamma_qg(c, nf);
    let qph = gamma_qph(c, nf);
    let nsp = gamma_nsp(c, nf);
    [
        [
            e2_tot * gamma_gg(c, nf),
            e2_tot * gamma_gph(c, nf),
            cc.e2avg() * gq,
            cc.vue2m() * gq,
        ],
        [
            e2_tot * gamma_phg(c, nf),
            gamma_phph(c, nf),
            cc.e2avg() * phq,
            cc.vue2m() * phq,
        ],
        [
            cc.e2avg() * qg,
            cc.e2avg() * qph,
            cc.e2avg() * nsp,
            cc.vue2m() * nsp,
        ],
        [
            cc.vde2m() * qg,
            cc.vde2m() * qph,
            cc.vde2m() * nsp,
            cc.e2delta() * nsp,
        ],
    ]
}

/// Compute the valence anomalous dimension matrix.
pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let cc = ChargeCombinations { nf };
    let g = gamma_nsm(c, nf);
    [
        [cc.e2avg() * g, cc.vue2m() * g],
        [cc.vde2m() * g, cc.e2delta() * g],
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
            let cc = ChargeCombinations { nf };
            assert_approx_eq_cmplx!(
                f64,
                EU2 * gamma_qg(&mut c, cc.nu())
                    + ED2 * gamma_qg(&mut c, cc.nd())
                    + (cc.nu() as f64 * EU2 + cc.nd() as f64 * ED2) * gamma_phg(&mut c, nf)
                    + (cc.nu() as f64 * EU2 + cc.nd() as f64 * ED2) * gamma_gg(&mut c, nf),
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
            let cc = ChargeCombinations { nf };
            assert_approx_eq_cmplx!(
                f64,
                EU2 * gamma_qph(&mut c, cc.nu())
                    + ED2 * gamma_qph(&mut c, cc.nd())
                    + gamma_phph(&mut c, nf)
                    + (cc.nu() as f64 * EU2 + cc.nd() as f64 * ED2) * gamma_gph(&mut c, nf),
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

    #[test]
    fn test_g3_shift() {
        for N in [cmplx!(1.23, 0.), cmplx!(5., 0.), cmplx!(2., 1.)] {
            let mut c = Cache::new(N);
            let mut c2 = Cache::new(N + 2.);
            assert_approx_eq_cmplx!(
                f64,
                g3_shift(&mut c),
                c2.get(K::G3) - c.get(K::G3),
                epsilon = 1e-6
            );
        }
    }
}
