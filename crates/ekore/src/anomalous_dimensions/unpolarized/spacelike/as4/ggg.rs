/// Compute the singlet gluon-to-gluon anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::constants::{ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{
    lm11, lm11m1, lm11m2, lm12m1, lm12m2, lm13m1, lm13m2, lm14m1,
};

// The routine is taken from [\[Falcioni:2024qpd\]][crate::bib:Falcioni:2024qpd].
pub fn gamma_gg(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    let nf2 = nf.pow(2) as f64;
    let nf3 = nf.pow(3) as f64;

    // The known large-x coefficients [except delta(1-x)]
    let A4gluon = 40880.330 - 11714.246 * nf as f64 + 440.04876 * nf2 + 7.3627750 * nf3;
    let mut B4gluon = 68587.64 - 18143.983 * nf as f64 + 423.81135 * nf2 + 9.0672154 * 0.1 * nf3;

    // The coefficient of delta(1-x), also called the virtual anomalous
    // dimension. nf^0 and nf^1 are still approximate, but the error at
    // nf^1 is far too small to be relevant in this context.
    if variation == 1 {
        B4gluon -= 0.2;
    } else if variation == 2 {
        B4gluon += 0.2;
    }

    let Ccoeff = 8.5814120 * f64::pow(10., 4) - 1.3880515 * f64::pow(10., 4) * nf as f64
        + 1.3511111 * f64::pow(10., 2) * nf2;
    let Dcoeff = 5.4482808 * f64::pow(10., 4)
        - 4.3411337 * f64::pow(10., 3) * nf as f64
        - 2.1333333 * 10. * nf2;

    let x1L4cff = 5.6460905 * 10. * nf as f64 - 3.6213992 * nf2;
    let x1L3cff =
        2.4755054 * f64::pow(10., 2) * nf as f64 - 4.0559671 * 10. * nf2 + 1.5802469 * nf3;

    // The known coefficients of 1/x*ln^a x terms, a = 3,2
    let bfkl0 = -8.3086173 * f64::pow(10., 3);
    let bfkl1 = -1.0691199 * f64::pow(10., 5) - 9.9638304 * f64::pow(10., 2) * nf as f64;

    let x0L6cff = 1.44 * f64::pow(10., 2) - 2.7786008 * 10. * nf as f64 + 7.9012346 * 0.1 * nf2;
    let x0L5cff =
        -1.44 * f64::pow(10., 2) - 1.6208066 * f64::pow(10., 2) * nf as f64 + 1.4380247 * 10. * nf2;
    let x0L4cff = 2.6165784 * f64::pow(10., 4) - 3.3447551 * f64::pow(10., 3) * nf as f64
        + 9.1522635 * 10. * nf2
        - 1.9753086 * 0.1 * nf3;

    // The resulting part of the function
    let P3gg01 = bfkl0 * (-(6. / (-1. + n).powu(4)))
        + bfkl1 * 2. / (-1. + n).powu(3)
        + x0L6cff * 720. / n.powu(7)
        + x0L5cff * -120. / n.powu(6)
        + x0L4cff * 24. / n.powu(5)
        + A4gluon * (-S1)
        + B4gluon
        + Ccoeff * lm11(n, S1)
        + Dcoeff * 1. / n
        + x1L4cff * lm14m1(n, S1, S2, S3, S4)
        + x1L3cff * lm13m1(n, S1, S2, S3);

    // The selected approximations for nf = 3, 4, 5
    let P3ggApp1;
    let P3ggApp2;
    if nf == 3 {
        P3ggApp1 = P3gg01
            - 421311.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 325557.0 * 1. / ((-1. + n) * n)
            + 1679790.0 * (1. / (n + n.powu(2)))
            - 1456863.0 * (1. / (2. + 3. * n + n.powu(2)))
            + 3246307.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 2026324.0 * 2. / n.powu(3)
            + 549188.0 * (-(6. / n.powu(4)))
            + 8337.0 * lm11m1(n, S1)
            + 26718.0 * lm12m1(n, S1, S2)
            - 27049.0 * lm13m2(n, S1, S2, S3);
        P3ggApp2 = P3gg01
            - 700113.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 2300581.0 * 1. / ((-1. + n) * n)
            + 896407.0 * (1. / n - n / (2. + 3. * n + n.powu(2)))
            - 162733.0 * (1. / (6. + 5. * n + n.powu(2)))
            - 2661862.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 196759.0 * 2. / n.powu(3)
            - 260607.0 * (-(6. / n.powu(4)))
            + 84068.0 * lm11m1(n, S1)
            + 346318.0 * lm12m1(n, S1, S2)
            + 315725.0
                * (-3. * S1.powu(2) + 6. * n * S1 * (ZETA2 - S2)
                    - 3. * (S2 + 2. * n * (S3 - ZETA3)))
                / (3. * n.powu(2));
    } else if nf == 4 {
        P3ggApp1 = P3gg01
            - 437084.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 361570.0 * 1. / ((-1. + n) * n)
            + 1696070.0 * (1. / (n + n.powu(2)))
            - 1457385.0 * (1. / (2. + 3. * n + n.powu(2)))
            + 3195104.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 2009021.0 * 2. / n.powu(3)
            + 544380.0 * (-(6. / n.powu(4)))
            + 9938.0 * lm11m1(n, S1)
            + 24376.0 * lm12m1(n, S1, S2)
            - 22143.0 * lm13m2(n, S1, S2, S3);
        P3ggApp2 = P3gg01
            - 706649.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 2274637.0 * 1. / ((-1. + n) * n)
            + 836544.0 * (1. / n - n / (2. + 3. * n + n.powu(2)))
            - 199929.0 * (1. / (6. + 5. * n + n.powu(2)))
            - 2683760.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 168802.0 * 2. / n.powu(3)
            - 250799.0 * (-(6. / n.powu(4)))
            + 36967.0 * lm11m1(n, S1)
            + 24530.0 * lm12m1(n, S1, S2)
            - 71470.0 * lm12m2(n, S1, S2);
    } else if nf == 5 {
        P3ggApp1 = P3gg01
            - 439426.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 293679.0 * 1. / ((-1. + n) * n)
            + 1916281.0 * (1. / (n + n.powu(2)))
            - 1615883.0 * (1. / (2. + 3. * n + n.powu(2)))
            + 3648786.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 2166231.0 * 2. / n.powu(3)
            + 594588.0 * (-(6. / n.powu(4)))
            + 50406.0 * lm11m1(n, S1)
            + 24692.0 * lm12m1(n, S1, S2)
            + 174067.0 * lm11m2(n, S1);
        P3ggApp2 = P3gg01
            - 705978.0 * (-(1. / (-1. + n).powu(2)) + 1. / n.powu(2))
            - 2192234.0 * 1. / ((-1. + n) * n)
            + 1730508.0 * (1. / (2. + 3. * n + n.powu(2)))
            + 353143.0
                * ((12. + 9. * n + n.powu(2))
                    / (6. * n + 11. * n.powu(2) + 6. * n.powu(3) + n.powu(4)))
            - 2602682.0 * (-(1. / n.powu(2)) + 1. / (1. + n).powu(2))
            + 178960.0 * 2. / n.powu(3)
            - 218133.0 * (-(6. / n.powu(4)))
            + 2285.0 * lm11m1(n, S1)
            + 19295.0 * lm12m1(n, S1, S2)
            - 13719.0 * lm12m2(n, S1, S2);
    } else {
        panic!("nf=6 is not available at N3LO");
    }

    // We return (for now) one of the two error-band representatives
    // or the present best estimate, their average
    let P3GGA = Complex::from(match variation {
        1 => P3ggApp1,
        2 => P3ggApp2,
        _ => 0.5 * (P3ggApp1 + P3ggApp2),
    });
    -P3GGA
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_reference_moments() {
        fn gg3_moment(N: usize, nf: f64) -> f64 {
            let nf2 = nf * nf;
            let nf3 = nf2 * nf;
            let mom_list = [
                654.4627782205557 * nf - 245.6106197887179 * nf2 + 0.9249909688301847 * nf3,
                39876.123276008046 - 10103.4511350227 * nf
                    + 437.0988475397789 * nf2
                    + 12.955565459350593 * nf3,
                53563.84353419538 - 14339.131035160317 * nf
                    + 652.7773306808972 * nf2
                    + 16.654103652963503 * nf3,
                62279.7437813437 - 17150.696783851945 * nf
                    + 785.8806126875509 * nf2
                    + 18.933103109772713 * nf3,
            ];
            mom_list[(N - 2) / 2]
        }
        for variation in [0, 1, 2] {
            for NF in [3, 4, 5] {
                for N in [2.0, 4.0, 6.0, 8.0] {
                    let mut c = Cache::new(cmplx!(N, 0.));
                    let test_value = gamma_gg(&mut c, NF, variation);
                    assert_approx_eq_cmplx!(
                        f64,
                        test_value,
                        cmplx!(gg3_moment(N as usize, NF as f64), 0.),
                        rel = 4e-4
                    );
                }
            }
        }
    }
}
