/// Compute the non-singlet-like non-singlet anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::cmplx;
use crate::constants::{CF, ZETA3};
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11, lm11m1, lm12m1, lm13m1};

/// Compute the valence-like non-singlet anomalous dimension.
///
/// The routine is taken from [\[Moch:2017uml\]][crate::bib::Moch2017uml].
///
/// The $n_f^{0,1}$ leading large-$N_c$ contributions and the $n_f^2$ part
/// are high-accuracy (0.1% or better) parametrizations of the exact
/// results. The $n_f^3$ expression is exact up to numerical truncations.
///
/// The remaining $n_f^{0,1}$ terms are approximations based on the first
/// eight even moments together with small-x and large-x constraints.
/// The two sets spanning the error estimate are called via `variation = 1`
/// and  ``variation = 2``. Any other value of `variation` invokes their average.
pub fn gamma_nsm(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);

    // Leading large-n_c, nf^0 and nf^1, parametrized
    #[rustfmt::skip]
    let P3NSA0 = 360.0 / n.powu(7)
        - 1920.0 / n.powu(6)
        + 7147.812 / n.powu(5)
        - 17179.356 / n.powu(4)
        + 34241.9 / n.powu(3)
        - 51671.329999999994 / n.powu(2)
        + 19069.8 * lm11(n, S1)
        - (491664.8019540468 / n)
        - 4533.0 / (1. + n).powu(3)
        - 11825.0 / (1. + n).powu(2)
        + 129203.0 / (1. + n)
        - 254965.0 / (2. + n)
        + 83377.5 / (3. + n)
        - 45750.0 / (4. + n)
        + (49150.0 * (6.803662258392675 + n) * S1) / (n.powu(2) * (1.0 + n))
        + (334400.0 * S2) / n;
    #[rustfmt::skip]
    let P3NSA1 = 160.0 / n.powu(6)
        - 864.0 / n.powu(5)
        + 2583.1848 / n.powu(4)
        - 5834.624 / n.powu(3)
        + 9239.374 / n.powu(2)
        - 3079.76 * lm11(n, S1)
        - (114047.0 / n)
        - 465.0 / (1. + n).powu(4)
        - 1230.0 / (1. + n).powu(3)
        + 7522.5 / (1. + n).powu(2)
        + 55669.3 / (1. + n)
        - 43057.8 / (2. + n)
        + 13803.8 / (3. + n)
        - 7896.0 / (4. + n)
        - (120.0 * (-525.063 + n) * S1) / (n.powu(2) * (1.0 + n))
        + (63007.5 * S2) / n;

    // Nonleading large-n_c, nf^0 and nf^1: two approximations
    #[rustfmt::skip]
    let P3NMA01 = 0.4964335 * (720. / n.powu(7) - 720.0 / n.powu(6))
        - 13.5288 / n.powu(4)
        + 1618.07 / n.powu(2)
        - 2118.8669999999997 * lm11(n, S1)
        + 31897.8 * lm11m1(n, S1)
        + 4653.76 * lm12m1(n, S1, S2)
        + 3902.3590000000004 / n
        + 5992.88 / (1. + n)
        + 19335.7 / (2. + n)
        - 31321.4 / (3. + n);
    #[rustfmt::skip]
    let P3NMA02 = 0.4964335 * (720. / n.powu(7) - 2160.0 / n.powu(6))
        - 189.6138 / n.powu(4)
        + 3065.92 / n.powu(3)
        - 2118.8669999999997 * lm11(n, S1)
        - 3997.39 * lm11m1(n, S1)
        + 511.567 * lm13m1(n, S1, S2, S3)
        - (2099.268 / n)
        + 4043.59 / (1. + n)
        - 19430.190000000002 / (2. + n)
        + 15386.6 / (3. + n);

    #[rustfmt::skip]
    let P3NMA11 = 64.7083 / n.powu(5)
        - 254.024 / n.powu(3)
        + 337.931 * lm11(n, S1)
        + 1856.63 * lm11m1(n, S1)
        + 440.17 * lm12m1(n, S1, S2)
        + 419.53485 / n
        + 114.457 / (1. + n)
        + 2341.816 / (2. + n)
        - 2570.73 / (3. + n);

    #[rustfmt::skip]
    let P3NMA12 = -17.0616 / n.powu(6)
        - 19.53254 / n.powu(3)
        + 337.931 * lm11(n, S1)
        - 1360.04 * lm11m1(n, S1)
        + 38.7337 * lm13m1(n, S1, S2, S3)
        - (367.64646999999997 / n)
        + 335.995 / (1. + n)
        - 1269.915 / (2. + n)
        + 1605.91 / (3. + n);

    // nf^2 (parametrized) and nf^3 (exact)
    #[rustfmt::skip]
    let P3NSMA2 = -(
        -193.84583328013258
        - 23.7037032 / n.powu(5)
        + 117.5967 / n.powu(4)
        - 256.5896 / n.powu(3)
        + 437.881 / n.powu(2)
        + 720.385709813466 / n
        - 48.720000000000006 / (1. + n).powu(4)
        + 189.51000000000002 / (1. + n).powu(3)
        + 391.02500000000003 / (1. + n).powu(2)
        + 367.4750000000001 / (1. + n)
        + 404.47249999999997 / (2. + n)
        - 2063.325 / ((1. + n).powu(2) * (2. + n))
        - (1375.55 * n) / ((1. + n).powu(2) * (2. + n))
        + 687.775 / ((1. + n) * (2. + n))
        - 81.71999999999998 / (3. + n)
        + 114.9225 / (4. + n)
        + 195.5772 * S1
        - (817.725 * S1) / n.powu(2)
        + (714.46361 * S1) / n
        - (687.775 * S1) / (1. + n)
        - (817.725 * S2) / n
    );
    let eta = 1. / n * 1. / (n + 1.);
    #[rustfmt::skip]
    let P3NSA3 = -CF * (
        -32. / 27. * ZETA3 * eta
        - 16. / 9. * ZETA3
        - 16. / 27. * eta.powu(4)
        - 16. / 81. * eta.powu(3)
        + 80. / 27. * eta.powu(2)
        - 320. / 81. * eta
        + 32. / 27. * 1. / (n + 1.).powu(4)
        + 128. / 27. * 1. / (n + 1.).powu(2)
        + 64. / 27. * S1 * ZETA3
        - 32. / 81. * S1
        - 32. / 81. * S2
        - 160. / 81. * S3
        + 32. / 27. * S4
        + 131. / 81.
    );

    let mut result = cmplx!(0., 0.);

    // Assembly regular piece.
    let nf = nf as f64;
    let P3NSMAI = P3NSA0 + nf * P3NSA1 + nf.pow(3) * P3NSA3 + nf.pow(2) * P3NSMA2;
    result += match variation {
        1 => P3NSMAI + P3NMA01 + nf * P3NMA11,
        2 => P3NSMAI + P3NMA02 + nf * P3NMA12,
        _ => P3NSMAI + 0.5 * ((P3NMA01 + P3NMA02) + nf * (P3NMA11 + P3NMA12)),
    };

    // The singular piece.
    let A4qI = 2.120902 * f64::pow(10., 4) - 5.179372 * f64::pow(10., 3) * nf
        // + 1.955772 * f64::pow(10.,2) * nf.pow(2)
        // + 3.272344 * nf.pow(3)
    ;
    let A4ap1 = -511.228 + 7.08645 * nf;
    let A4ap2 = -502.481 + 7.82077 * nf;
    let D1 = 1. / n - S1;
    result += match variation {
        1 => (A4qI + A4ap1) * D1,
        2 => (A4qI + A4ap2) * D1,
        _ => (A4qI + 0.5 * (A4ap1 + A4ap2)) * D1,
    };

    // ..The local piece.
    let B4qI = 2.579609 * f64::pow(10., 4) + 0.08 - (5.818637 * f64::pow(10., 3) + 0.97) * nf
        // + (1.938554 * f64::pow(10.,2) + 0.0037) * nf.pow(2)
        // + 3.014982 * nf.pow(3)
    ;
    let B4ap1 = -2426.05 + 266.674 * nf - 0.05 * nf;
    let B4ap2 = -2380.255 + 270.518 * nf - 0.05 * nf;
    result += match variation {
        1 => B4qI + B4ap1,
        2 => B4qI + B4ap2,
        _ => B4qI + 0.5 * (B4ap1 + B4ap2),
    };

    -result
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_quark_number_conservation() {
        let NF = 5;
        let mut c = Cache::new(cmplx!(1., 0.));
        let refs: [f64; 3] = [0.06776363, 0.064837, 0.07069];
        for var in [0, 1, 2] {
            let test_value = gamma_nsm(&mut c, NF, var);
            assert_approx_eq_cmplx!(f64, test_value, cmplx!(refs[var as usize], 0.), rel = 6e-5);
        }
    }

    #[test]
    fn test_reference_moments() {
        let NF = 4;
        let nsm_nf4_refs: [f64; 7] = [
            4322.890485339998,
            5491.581109692005,
            6221.256799360004,
            6774.606221595994,
            7229.056043916002,
            7618.358743427995,
            7960.658678124,
        ];
        for N in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] {
            let mut c = Cache::new(cmplx!(N, 0.));
            let test_value = gamma_nsm(&mut c, NF, 0);
            assert_approx_eq_cmplx!(
                f64,
                test_value,
                cmplx!(nsm_nf4_refs[((N - 2.) / 2.) as usize], 0.),
                rel = 8e-5
            );
        }
    }
}
