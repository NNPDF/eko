//! NNLO QCD

use ::num::complex::Complex;
use num::traits::Pow;

use crate::cmplx;
use crate::constants::{ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};

/// Compute the valence-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.8) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsm(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);

    #[rustfmt::skip]
    let pm2 =
        -1174.898 * (S1 - 1.0 / N)
        + 1295.470
        - 714.1 * S1 / N
        - 433.2 / (N + 3.0)
        + 297.0 / (N + 2.0)
        - 3505.0 / (N + 1.0)
        + 1860.2 / N
        - 1465.2 / N.powu(2)
        + 399.2 * 2.0 / N.powu(3)
        - 320.0 / 9.0 * 6.0 / N.powu(4)
        + 116.0 / 81.0 * 24.0 / N.powu(5)
        + 684.0 * E1
        + 251.2 * E2;

    #[rustfmt::skip]
    let pm2_nf =
        183.187 * (S1 - 1.0 / N)
        - 173.933
        + 5120./ 81.0 * S1 / N
        + 34.76 / (N + 3.0)
        + 77.89 / (N + 2.0)
        + 406.5 / (N + 1.0)
        - 216.62 / N
        + 172.69 / N.powu(2)
        - 3216.0 / 81.0 * 2.0 / N.powu(3)
        + 256.0 / 81.0 * 6.0 / N.powu(4)
        - 65.43 * E1
        + 1.136 * 6.0 / (N + 1.0).powu(4);

    #[rustfmt::skip]
    let pf2_nfnf =
        -(
            17.0 / 72.0
            - 2.0 / 27.0 * S1
            - 10.0 / 27.0 * S2
            + 2.0 / 9.0 * S3
            - (12.0 * N.powu(4) + 2.0 * N.powu(3) - 12.0 * N.powu(2) - 2.0 * N + 3.0)
            / (27.0 * N.powu(3) * (N + 1.0).powu(3))
        )* 32.0 / 3.0;

    let result = pm2 + (nf as f64) * pm2_nf + (nf as f64).pow(2) * pf2_nfnf;
    -1. * result
}

/// Compute the singlet-like non-singlet anomalous dimension.
///
/// Implements Eq. (3.7) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsp(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);

    #[rustfmt::skip]
    let pp2 =
        -1174.898 * (S1 - 1.0 / N)
        + 1295.384
        - 714.1 * S1 / N
        - 522.1 / (N + 3.0)
        + 243.6 / (N + 2.0)
        - 3135.0 / (N + 1.0)
        + 1641.1 / N
        - 1258.0 / N.powu(2)
        + 294.9 * 2.0 / N.powu(3)
        - 800. / 27.0 * 6.0 / N.powu(4)
        + 128. / 81.0 * 24.0 / N.powu(5)
        + 563.9 * E1
        + 256.8 * E2;

    #[rustfmt::skip]
    let pp2_nf =
        183.187 * (S1 - 1.0 / N)
        - 173.924
        + 5120. / 81.0 * S1 / N
        + 44.79 / (N + 3.0)
        + 72.94 / (N + 2.0)
        + 381.1 / (N + 1.0)
        - 197.0 / N
        + 152.6 / N.powu(2)
        - 2608.0 / 81.0 * 2.0 / N.powu(3)
        + 192.0 / 81.0 * 6.0 / N.powu(4)
        - 56.66 * E1
        + 1.497 * 6.0 / (N + 1.0).powu(4);

    #[rustfmt::skip]
    let pf2_nfnf =
        -(
            17.0 / 72.0
            - 2.0 / 27.0 * S1
            - 10.0 / 27.0 * S2
            + 2.0 / 9.0 * S3
            - (12.0 * N.powu(4) + 2.0 * N.powu(3) - 12.0 * N.powu(2) - 2.0 * N + 3.0)
            / (27.0 * N.powu(3) * (N + 1.0).powu(3))
        )* 32.0/ 3.0;

    let result = pp2 + (nf as f64) * pp2_nf + (nf as f64).pow(2) * pf2_nfnf;
    -1. * result
}

/// Compute the valence non-singlet anomalous dimension.
///
/// Implements Eq. (3.9) of [\[Moch:2004pa\]][crate::bib::Moch2004pa].
pub fn gamma_nsv(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);
    let B11 = -(S1 + 1.0 / (N + 1.0)) / (N + 1.0);
    let B12 = -(S1 + 1.0 / (N + 1.0) + 1.0 / (N + 2.0)) / (N + 2.0);

    let B1M = if N.im.abs() < 1.0e-5 && (N - 1.0).re.abs() < 1.0e-5 {
        cmplx![-ZETA2, 0.]
    } else {
        -(S1 - 1.0 / N) / (N - 1.0)
    };

    #[rustfmt::skip]
    let ps2 = -(
        -163.9 * (B1M + S1 / N)
        - 7.208 * (B11 - B12)
        + 4.82 * (1.0 / (N + 3.0) - 1.0 / (N + 4.0))
        - 43.12 * (1.0 / (N + 2.0) - 1.0 / (N + 3.0))
        + 44.51 * (1.0 / (N + 1.0) - 1.0 / (N + 2.0))
        + 151.49 * (1.0 / N - 1.0 / (N + 1.0))
        - 178.04 / N.powu(2)
        + 6.892 * 2.0 / N.powu(3)
        - 40.0 / 27.0 * (-2.0 * 6.0 / N.powu(4) - 24.0 / N.powu(5))
        - 173.1 * E1
        + 46.18 * E2
    );

    gamma_nsm(c, nf) + (nf as f64) * ps2
}

/// Compute the pure-singlet quark-quark anomalous dimension.
///
/// Implements Eq. (3.10) of [\[Vogt:2004mw\]][crate::bib::Vogt2004mw].
pub fn gamma_ps(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);

    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E11 = (S1 + 1.0 / (N + 1.0)) / (N + 1.0).powu(2)
        + (S2 + 1.0 / (N + 1.0).powu(2) - ZETA2) / (N + 1.0);
    let B21 = ((S1 + 1.0 / (N + 1.0)).powu(2) + S2 + 1.0 / (N + 1.0).powu(2)) / (N + 1.0);
    let B3 = -(S1.powu(3) + 3.0 * S1 * S2 + 2.0 * S3) / N;

    #[rustfmt::skip]
    let B31 = -(
        (S1 + 1.0 / (N + 1.0)).powu(3)
        + 3.0 * (S1 + 1.0 / (N + 1.0)) * (S2 + 1.0 / (N + 1.0).powu(2))
        + 2.0 * (S3 + 1.0 / (N + 1.0).powu(3))
    ) / (N + 1.0);

    #[rustfmt::skip]
    let ps1 =
        -3584.0 / 27.0 * (-1.0 / (N - 1.0).powu(2) + 1.0 / N.powu(2))
        - 506.0 * (1.0 / (N - 1.0) - 1.0 / N)
        + 160.0 / 27.0 * (24.0 / N.powu(5) - 24.0 / (N + 1.0).powu(5))
        - 400.0 / 9.0 * (-6.0 / N.powu(4) + 6.0 / (N + 1.0).powu(4))
        + 131.4 * (2.0 / N.powu(3) - 2.0 / (N + 1.0).powu(3))
        - 661.6 * (-1.0 / N.powu(2) + 1.0 / (N + 1.0).powu(2))
        - 5.926 * (B3 - B31)
        - 9.751 * ((S1.powu(2) + S2) / N - B21)
        - 72.11 * (-S1 / N + (S1 + 1.0 / (N + 1.0)) / (N + 1.0))
        + 177.4 * (1.0 / N - 1.0 / (N + 1.0))
        + 392.9 * (1.0 / (N + 1.0) - 1.0 / (N + 2.0))
        - 101.4 * (1.0 / (N + 2.0) - 1.0 / (N + 3.0))
        - 57.04 * (E1 - E11);

    #[rustfmt::skip]
    let ps2 =
        256.0 / 81.0 * (1.0 / (N - 1.0) - 1.0 / N)
        + 32.0 / 27.0 * (-6.0 / N.powu(4) + 6.0 / (N + 1.0).powu(4))
        + 17.89 * (2.0 / N.powu(3) - 2.0 / (N + 1.0).powu(3))
        + 61.75 * (-1.0 / N.powu(2) + 1.0 / (N + 1.0).powu(2))
        + 1.778 * ((S1.powu(2) + S2) / N - B21)
        + 5.944 * (-S1 / N + (S1 + 1.0 / (N + 1.0)) / (N + 1.0))
        + 100.1 * (1.0 / N - 1.0 / (N + 1.0))
        - 125.2 * (1.0 / (N + 1.0) - 1.0 / (N + 2.0))
        + 49.26 * (1.0 / (N + 2.0) - 1.0 / (N + 3.0))
        - 12.59 * (1.0 / (N + 3.0) - 1.0 / (N + 4.0))
        - 1.889 * (E1 - E11);

    let result = (nf as f64) * ps1 + (nf as f64).pow(2) * ps2;
    -1.0 * result
}

/// Compute the quark-gluon singlet anomalous dimension.
///
/// Implements Eq. (3.11) of [\[Vogt:2004mw\]][crate::bib::Vogt2004mw].
pub fn gamma_qg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);

    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);
    let B3 = -(S1.powu(3) + 3.0 * S1 * S2 + 2.0 * S3) / N;
    let B4 = (S1.powu(4) + 6.0 * S1.powu(2) * S2 + 8.0 * S1 * S3 + 3.0 * S2.powu(2) + 6.0 * S4) / N;

    #[rustfmt::skip]
    let qg1 =
        896.0 / 3.0 / (N - 1.0).powu(2)
        - 1268.3 / (N - 1.0)
        + 536.0 / 27.0 * 24.0 / N.powu(5)
        + 44.0 / 3.0 * 6.0 / N.powu(4)
        + 881.5 * 2.0 / N.powu(3)
        - 424.9 / N.powu(2)
        + 100.0 / 27.0 * B4
        - 70.0 / 9.0 * B3
        - 120.5 * (S1.powu(2) + S2) / N
        - 104.42 * S1 / N
        + 2522.0 / N
        - 3316.0 / (N + 1.0)
        + 2126.0 / (N + 2.0)
        + 1823.0 * E1
        - 25.22 * E2
        + 252.5 * 6.0 / (N + 1.0).powu(4);

    #[rustfmt::skip]
    let qg2 =
        1112.0 / 243.0 / (N - 1.0)
        - 16.0 / 9.0 * 24.0 / N.powu(5)
        + 376.0 / 27.0 * 6.0 / N.powu(4)
        - 90.8 * 2.0 / N.powu(3)
        + 254.0 / N.powu(2)
        + 20.0 / 27.0 * B3
        + 200.0 / 27.0 * (S1.powu(2) + S2) / N
        + 5.496 * S1 / N
        - 252.0 / N
        + 158.0 / (N + 1.0)
        + 145.4 / (N + 2.0)
        - 139.28 / (N + 3.0)
        - 53.09 * E1
        - 80.616 * E2
        - 98.07 * 2.0 / (N + 1.0).powu(3)
        - 11.70 * 6.0 / (N + 1.0).powu(4);

    let result = (nf as f64) * qg1 + (nf as f64).pow(2) * qg2;
    -1.0 * result
}

/// Compute the gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (3.12) of [\[Vogt:2004mw\]][crate::bib::Vogt2004mw].
pub fn gamma_gq(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);

    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);
    let B21 = ((S1 + 1.0 / (N + 1.0)).powu(2) + S2 + 1.0 / (N + 1.0).powu(2)) / (N + 1.0);
    let B3 = -(S1.powu(3) + 3.0 * S1 * S2 + 2.0 * S3) / N;
    let B4 = (S1.powu(4) + 6.0 * S1.powu(2) * S2 + 8.0 * S1 * S3 + 3.0 * S2.powu(2) + 6.0 * S4) / N;

    #[rustfmt::skip]
    let gq0 =
        -1189.3 * 1.0 / (N - 1.0).powu(2)
        + 6163.1 / (N - 1.0)
        - 4288.0 / 81.0 * 24.0 / N.powu(5)
        - 1568.0 / 9.0 * 6.0 / N.powu(4)
        - 1794.0 * 2.0 / N.powu(3)
        - 4033.0 * 1.0 / N.powu(2)
        + 400.0 / 81.0 * B4
        + 2200.0 / 27.0 * B3
        + 606.3 * (S1.powu(2) + S2) / N
        - 2193.0 * S1 / N
        - 4307.0 / N
        + 489.3 / (N + 1.0)
        + 1452.0 / (N + 2.0)
        + 146.0 / (N + 3.0)
        - 447.3 * E2
        - 972.9 * 2.0 / (N + 1.0).powu(3);

    #[rustfmt::skip]
    let gq1 =
        -71.082 / (N - 1.0).powu(2)
        - 46.41 / (N - 1.0)
        + 128.0 / 27.0 * 24.0 / N.powu(5)
        - 704. / 81.0 * 6.0 / N.powu(4)
        + 20.39 * 2.0 / N.powu(3)
        - 174.8 * 1.0 / N.powu(2)
        - 400.0 / 81.0 * B3
        - 68.069 * (S1.powu(2) + S2) / N
        + 296.7 * S1 / N
        - 183.8 / N
        + 33.35 / (N + 1.0)
        - 277.9 / (N + 2.0)
        + 108.6 * 2.0 / (N + 1.0).powu(3)
        - 49.68 * E1;

    #[rustfmt::skip]
    let gq2 = (
        64.0 * (-1.0 / (N - 1.0) + 1.0 / N + 2.0 / (N + 1.0))
        + 320.0
        * (
            -(S1 - 1.0 / N) / (N - 1.0)
            + S1 / N
            - 0.8 * (S1 + 1.0 / (N + 1.0)) / (N + 1.0)
        )
        + 96.0
        * (
            ((S1 - 1.0 / N).powu(2) + S2 - 1.0 / N.powu(2)) / (N - 1.0)
            - (S1.powu(2) + S2) / N
            + 0.5 * B21
        )
    ) / 27.0;

    let result = gq0 + (nf as f64) * gq1 + (nf as f64).pow(2) * gq2;
    -1.0 * result
}

/// Compute the gluon-quark singlet anomalous dimension.
///
/// Implements Eq. (3.13) of [\[Vogt:2004mw\]][crate::bib::Vogt2004mw].
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n;
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);

    let E1 = S1 / N.powu(2) + (S2 - ZETA2) / N;
    let E11 = (S1 + 1.0 / (N + 1.0)) / (N + 1.0).powu(2)
        + (S2 + 1.0 / (N + 1.0).powu(2) - ZETA2) / (N + 1.0);
    let E2 = 2.0 * (-S1 / N.powu(3) + (ZETA2 - S2) / N.powu(2) - (S3 - ZETA3) / N);

    #[rustfmt::skip]
    let gg0 =
        -2675.8 / (N - 1.0).powu(2)
        + 14214.0 / (N - 1.0)
        - 144.0 * 24.0 / N.powu(5)
        - 72.0 * 6.0 / N.powu(4)
        - 7471.0 * 2.0 / N.powu(3)
        - 274.4 / N.powu(2)
        - 20852.0 / N
        + 3968.0 / (N + 1.0)
        - 3363.0 / (N + 2.0)
        + 4848.0 / (N + 3.0)
        + 7305.0 * E1
        + 8757.0 * E2
        - 3589.0 * S1 / N
        + 4425.894
        - 2643.521 * (S1 - 1.0 / N);

    #[rustfmt::skip]
    let gg1 =
        -157.27 / (N - 1.0).powu(2)
        + 182.96 / (N - 1.0)
        + 512.0 / 27.0 * 24.0 / N.powu(5)
        - 832.0 / 9.0 * 6.0 / N.powu(4)
        + 491.3 * 2.0 / N.powu(3)
        - 1541.0 / N.powu(2)
        - 350.2 / N
        + 755.7 / (N + 1.0)
        - 713.8 / (N + 2.0)
        + 559.3 / (N + 3.0)
        + 26.15 * E1
        - 808.7 * E2
        + 320.0 * S1 / N
        - 528.723
        + 412.172 * (S1 - 1.0 / N);

    #[rustfmt::skip]
    let gg2 =
        -680.0 / 243.0 / (N - 1.0)
        + 32.0 / 27.0 * 6.0 / N.powu(4)
        + 9.680 * 2.0 / N.powu(3)
        + 3.422 / N.powu(2)
        - 13.878 / N
        + 153.4 / (N + 1.0)
        - 187.7 / (N + 2.0)
        + 52.75 / (N + 3.0)
        - 115.6 * E1
        + 85.25 * E11
        - 63.23 * E2
        + 6.4630
        + 16.0 / 9.0 * (S1 - 1.0 / N);

    let result = gg0 + (nf as f64) * gg1 + (nf as f64).pow(2) * gg2;
    -1.0 * result
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gamma_nsp(c, nf) + gamma_ps(c, nf);
    [
        [gamma_qq, gamma_qg(c, nf)],
        [gamma_gq(c, nf), gamma_gg(c, nf)],
    ]
}

#[cfg(test)]
mod tests {
    use crate::cmplx;
    use crate::{anomalous_dimensions::unpolarized::spacelike::as3::*, harmonics::cache::Cache};
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;

    const NF: u8 = 5;

    #[test]
    fn physical_constraints() {
        // number conservation
        let mut c = Cache::new(cmplx![1., 0.]);
        assert_approx_eq!(f64, gamma_nsv(&mut c, NF).re, -0.000960586, epsilon = 3e-7);
        assert_approx_eq!(f64, gamma_nsm(&mut c, NF).re, 0.000594225, epsilon = 6e-7);

        let mut c = Cache::new(cmplx![2., 0.]);
        let gS2 = gamma_singlet(&mut c, NF);
        // gluon momentum conservation
        assert_approx_eq!(f64, (gS2[0][1] + gS2[1][1]).re, -0.00388726, epsilon = 2e-6);
        // quark momentum conservation
        assert_approx_eq!(f64, (gS2[0][0] + gS2[1][0]).re, 0.00169375, epsilon = 2e-6);
    }

    #[test]
    fn N2() {
        let mut c = Cache::new(cmplx![2., 0.]);
        assert_approx_eq!(f64, gamma_nsv(&mut c, NF).re, 188.325593, epsilon = 3e-7);
    }
}
