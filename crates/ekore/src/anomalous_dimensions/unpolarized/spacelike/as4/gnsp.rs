/// Compute the singlet-like non-singlet anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::cmplx;
use crate::constants::{CF, ZETA3};
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11, lm11m1, lm12m1, lm13m1};

/// Compute the singlet-like non-singlet anomalous dimension.
///
/// See [gamma_nsm][super::gamma_nsm] for implementation details.
pub fn gamma_nsp(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
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
    let P3NPA01 = - 107.16 / n.powu(7)
        + 339.753 / n.powu(6)
        - 1341.01 / n.powu(5)
        + 2412.94 / n.powu(4)
        - 3678.88 / n.powu(3)
        - 2118.87 * lm11(n, S1)
        - 1777.27 * lm12m1(n, S1, S2)
        - 204.183 * lm13m1(n, S1, S2, S3)
        + 1853.56 / n
        - 8877.38 / (1. + n)
        + 7393.83 / (2. + n)
        - 2464.61 / (3. + n);
    #[rustfmt::skip]
    let P3NPA02 = - 107.16 / n.powu(7)
        + 339.753 / n.powu(6)
        - 1341.01 / n.powu(5)
        + 379.152 / n.powu(3)
        - 1389.73 / n.powu(2)
        - 2118.87 * lm11(n, S1)
        - 173.936 * lm12m1(n, S1, S2)
        + 223.078 * lm13m1(n, S1, S2, S3)
        - (2096.54 / n)
        + 8698.39 / (1. + n)
        - 19188.9 / (2. + n)
        + 10490.5 / (3. + n);
    #[rustfmt::skip]
    let P3NPA11 = -33.5802 / n.powu(6)
        + 111.802 / n.powu(5)
        + 50.772 / n.powu(4)
        - 118.608 / n.powu(3)
        + 337.931 * lm11(n, S1)
        - 143.813 * lm11m1(n, S1)
        - 18.8803 * lm13m1(n, S1, S2, S3)
        + 304.82503 / n
        - 1116.34 / (1. + n)
        + 2187.58 / (2. + n)
        - 1071.24 / (3. + n);
    #[rustfmt::skip]
    let P3NPA12 = - 33.5802 / n.powu(6)
        + 111.802 / n.powu(5)
        - 204.341 / n.powu(4)
        + 267.404 / n.powu(3)
        + 337.931 * lm11(n, S1)
        - 745.573 * lm11m1(n, S1)
        + 8.61438 * lm13m1(n, S1, S2, S3)
        - 385.52331999999996 / n
        + 690.151 / (1. + n)
        - 656.386 / (2. + n)
        + 656.386 / (3. + n);

    // nf^2 (parametrized) and nf^3 (exact)
    #[rustfmt::skip]
    let P3NSPA2 = -(
        -193.85906555742952
        - 18.962964 / n.powu(5)
        + 99.1605 / n.powu(4)
        - 225.141 / n.powu(3)
        + 393.0056000000001 / n.powu(2)
        - 403.50217685814835 / n
        - 34.425000000000004 / (1. + n).powu(4)
        + 108.42 / (1. + n).powu(3)
        - 93.8225 / (1. + n).powu(2)
        + 534.725 / (1. + n)
        + 246.50250000000003 / (2. + n)
        - 25.455 / ((1. + n).powu(2) * (2. + n))
        - (16.97 * n) / ((1. + n).powu(2) * (2. + n))
        + 8.485 / ((1. + n) * (2. + n))
        - 110.015 / (3. + n)
        + 78.9875 / (4. + n)
        + 195.5772 * S1
        - (101.0775 * S1) / n.powu(2)
        + (35.17361 * S1) / n
        - (8.485 * S1) / (1. + n)
        - (101.0775 * S2) / n
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
    let NF = nf as f64;
    let P3NSPAI = P3NSA0 + NF * P3NSA1 + NF.pow(2) * P3NSPA2 + NF.pow(3) * P3NSA3;
    result += match variation {
        1 => P3NSPAI + P3NPA01 + NF * P3NPA11,
        2 => P3NSPAI + P3NPA02 + NF * P3NPA12,
        _ => P3NSPAI + 0.5 * ((P3NPA01 + P3NPA02) + NF * (P3NPA11 + P3NPA12)),
    };

    // The singular piece.
    let A4qI = 2.120902 * f64::pow(10., 4) - 5.179372 * f64::pow(10., 3) * NF
        // + 1.955772 * f64::pow(10.,2) * NF.pow(2)
        // + 3.272344 * NF.pow(3)
    ;
    let A4ap1 = -507.152 + 7.33927 * NF;
    let A4ap2 = -505.209 + 7.53662 * NF;
    let D1 = 1. / n - S1;
    result += match variation {
        1 => (A4qI + A4ap1) * D1,
        2 => (A4qI + A4ap2) * D1,
        _ => (A4qI + 0.5 * (A4ap1 + A4ap2)) * D1,
    };

    // ..The local piece.
    let B4qI = 2.579609 * f64::pow(10., 4) + 0.08 - (5.818637 * f64::pow(10., 3) + 0.97) * NF
        // + (1.938554 * f64::pow(10.,2) + 0.0037) * NF.pow(2)
        // + 3.014982 * NF.pow(3)
    ;
    let B4ap1 = -2405.03 + 267.965 * NF;
    let B4ap2 = -2394.47 + 269.028 * NF;
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
    fn test_reference_moments() {
        let NF = 4;
        let nsp_nf4_refs: [f64; 8] = [
            3679.6690577439995,
            5066.339235808004,
            5908.005605364002,
            6522.700744595994,
            7016.383458928004,
            7433.340927783997,
            7796.397038483998,
            8119.044600816003,
        ];
        for N in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0] {
            let mut c = Cache::new(cmplx!(N, 0.));
            let test_value = gamma_nsp(&mut c, NF, 0);
            assert_approx_eq_cmplx!(
                f64,
                test_value,
                cmplx!(nsp_nf4_refs[((N - 2.) / 2.) as usize], 0.),
                rel = 4e-5
            );
        }
    }
}
