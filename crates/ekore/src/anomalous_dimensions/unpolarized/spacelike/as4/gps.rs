/// Compute the pure-singlet quark-to-quark anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11m1, lm12m1, lm12m2, lm13m1, lm13m2, lm14m1, lm14m2};

// The routine is taken from [\[Falcioni:2023luc\]][crate::bib:Falcioni:2023luc].
pub fn gamma_ps(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    let nf2 = nf.pow(2) as f64;
    let nf3 = nf.pow(3) as f64;
    let xm1lm1 = -(1. / (-1. + n).powu(2)) + 1. / n.powu(2);

    // Known large-x coefficients
    let x1L4cff = -5.6460905 * 10. * nf as f64 + 3.6213992 * nf2;
    let x1L3cff =
        -2.4755054 * f64::pow(10., 2) * nf as f64 + 4.0559671 * 10. * nf2 - 1.5802469 * nf3;
    let y1L4cff = -1.3168724 * 10. * nf as f64;
    let y1L3cff = -1.9911111 * f64::pow(10., 2) * nf as f64 + 1.3695473 * 10. * nf2;

    // Known small-x coefficients
    let bfkl1 = 1.7492273 * f64::pow(10., 3) * nf as f64;
    let x0L6cff = -7.5061728 * nf as f64 + 7.9012346 * 0.1 * nf2;
    let x0L5cff = 2.8549794 * 10. * nf as f64 + 3.7925926 * nf2;
    let x0L4cff =
        -8.5480010 * f64::pow(10., 2) * nf as f64 + 7.7366255 * 10. * nf2 - 1.9753086 * 0.1 * nf3;

    // The resulting part of the function
    let P3ps01 = bfkl1 * 2. / (-1. + n).powu(3)
        + x0L6cff * 720. / n.powu(7)
        + x0L5cff * -120. / n.powu(6)
        + x0L4cff * 24. / n.powu(5)
        + x1L3cff * lm13m1(n, S1, S2, S3)
        + x1L4cff * lm14m1(n, S1, S2, S3, S4)
        + y1L3cff * lm13m2(n, S1, S2, S3)
        + y1L4cff * lm14m2(n, S1, S2, S3, S4);

    // The selected approximations for nf = 3, 4, 5
    let P3psApp1: Complex<f64>;
    let P3psApp2: Complex<f64>;
    if nf == 3 {
        P3psApp1 = P3ps01 + 67731.0 * xm1lm1 + 274100.0 * 1. / ((-1. + n) * n)
            - 104493.0 * (1. / n - n / (2. + 3. * n + n.powu(2)))
            + 34403.0 * 1. / (6. + 5. * n + n.powu(2))
            + 353656.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 10620.0 * 2. / n.powu(3)
            + 40006.0 * -6. / n.powu(4)
            - 7412.1 * lm11m1(n, S1)
            - 2365.1 * lm12m1(n, S1, S2)
            + 1533.0 * lm12m2(n, S1, S2);
        P3psApp2 = P3ps01 + 54593.0 * xm1lm1 + 179748.0 * 1. / ((-1. + n) * n)
            - 195263.0 * 1. / (n + n.powu(2))
            + 12789.0 * 2. / (3. + 4. * n + n.powu(2))
            + 4700.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            - 103604.0 * 2. / n.powu(3)
            - 2758.3 * -6. / n.powu(4)
            - 2801.2 * lm11m1(n, S1)
            - 1986.9 * lm12m1(n, S1, S2)
            - 6005.9 * lm12m2(n, S1, S2);
    } else if nf == 4 {
        P3psApp1 = P3ps01 + 90154.0 * xm1lm1 + 359084.0 * 1. / ((-1. + n) * n)
            - 136319.0 * (1. / n - n / (2. + 3. * n + n.powu(2)))
            + 45379.0 * 1. / (6. + 5. * n + n.powu(2))
            + 461167.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 13869.0 * 2. / n.powu(3)
            + 52525.0 * -6. / n.powu(4)
            - 7498.2 * lm11m1(n, S1)
            - 2491.5 * lm12m1(n, S1, S2)
            + 1727.2 * lm12m2(n, S1, S2);
        P3psApp2 = P3ps01 + 72987.0 * xm1lm1 + 235802.0 * 1. / ((-1. + n) * n)
            - 254921.0 * 1. / (n + n.powu(2))
            + 17138.0 * 2. / (3. + 4. * n + n.powu(2))
            + 5212.9 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            - 135378.0 * 2. / n.powu(3)
            - 3350.9 * -6. / n.powu(4)
            - 1472.7 * lm11m1(n, S1)
            - 1997.2 * lm12m1(n, S1, S2)
            - 8123.3 * lm12m2(n, S1, S2);
    } else if nf == 5 {
        P3psApp1 = P3ps01 + 112481.0 * xm1lm1 + 440555.0 * 1. / ((-1. + n) * n)
            - 166581.0 * (1. / n - n / (2. + 3. * n + n.powu(2)))
            + 56087.0 * 1. / (6. + 5. * n + n.powu(2))
            + 562992.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 16882.0 * 2. / n.powu(3)
            + 64577.0 * -6. / n.powu(4)
            - 6570.1 * lm11m1(n, S1)
            - 2365.7 * lm12m1(n, S1, S2)
            + 1761.7 * lm12m2(n, S1, S2);
        P3psApp2 = P3ps01 + 91468.0 * xm1lm1 + 289658.0 * 1. / ((-1. + n) * n)
            - 311749.0 * 1. / (n + n.powu(2))
            + 21521.0 * 2. / (3. + 4. * n + n.powu(2))
            + 4908.9 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            - 165795.0 * 2. / n.powu(3)
            - 3814.9 * -6. / n.powu(4)
            + 804.5 * lm11m1(n, S1)
            - 1760.8 * lm12m1(n, S1, S2)
            - 10295.0 * lm12m2(n, S1, S2);
    } else {
        panic!("nf=6 is not available at N3LO");
    }
    // We return (for now) one of the two error-band boundaries
    // or the present best estimate, their average

    let P3psA = match variation {
        1 => P3psApp1,
        2 => P3psApp2,
        _ => 0.5 * (P3psApp1 + P3psApp2),
    };
    -P3psA
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    // Test the prediction of N=22 wrt to [\[Falcioni:2023luc\]][crate::bib:Falcioni:2023luc].
    fn test_gamma_ps_extrapolation() {
        let n22_ref: [f64; 3] = [6.2478570, 10.5202730, 15.6913948];
        let mut c = Cache::new(cmplx!(22., 0.));
        for NF in [3, 4, 5] {
            let test_value = gamma_ps(&mut c, NF, 0);
            assert_approx_eq_cmplx!(
                f64,
                test_value,
                cmplx!(n22_ref[(NF - 3) as usize], 0.),
                rel = 3e-5
            );
        }
    }

    #[test]
    fn test_reference_moments() {
        fn qq3ps_moment(N: usize, nf: f64) -> f64 {
            let nf2 = nf * nf;
            let nf3 = nf2 * nf;
            let mom_list = [
                -691.5937093082381 * nf + 84.77398149891167 * nf2 + 4.4669568492355864 * nf3,
                -109.33023358432462 * nf + 8.77688525974872 * nf2 + 0.3060771365698822 * nf3,
                -46.030613749542226 * nf + 4.744075766957513 * nf2 + 0.042548957282380874 * nf3,
                -24.01455020567638 * nf + 3.235193483272451 * nf2 - 0.007889256298951614 * nf3,
                -13.730393879922417 * nf + 2.3750187592472374 * nf2 - 0.02102924056123573 * nf3,
                -8.152592251923657 * nf + 1.8199581788320662 * nf2 - 0.024330231290833188 * nf3,
                -4.8404471801109565 * nf + 1.4383273806219803 * nf2 - 0.024479943136069916 * nf3,
                -2.7511363301137024 * nf + 1.164299642517469 * nf2 - 0.023546009234463816 * nf3,
                -1.375969240387974 * nf + 0.9608733183576097 * nf2 - 0.022264393374041958 * nf3,
                -0.4426815682220422 * nf + 0.8057453328332964 * nf2 - 0.02091826436475512 * nf3,
            ];
            mom_list[(N - 2) / 2]
        }
        for variation in [0, 1, 2] {
            for NF in [3, 4, 5] {
                for N in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0] {
                    let mut c = Cache::new(cmplx!(N, 0.));
                    let test_value = gamma_ps(&mut c, NF, variation);
                    assert_approx_eq_cmplx!(
                        f64,
                        test_value,
                        cmplx!(qq3ps_moment(N as usize, NF as f64), 0.),
                        rel = 4e-4
                    );
                }
            }
        }
    }
}
