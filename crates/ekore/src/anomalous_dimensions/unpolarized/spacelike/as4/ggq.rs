/// Compute the singlet quark-to-gluon anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::constants::ZETA2;
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11, lm12, lm12m1, lm13, lm14, lm14m1, lm15, lm15m1};

/// Compute the singlet quark-to-gluon anomalous dimension.
///
/// The routine is taken from [\[Falcioni:2025hfz\]][crate::bib::Falcioni2025hfz].
/// A previous version was given in [\[Falcioni:2024xyt\]][crate::bib::Falcioni2024xyt],
/// while a version based only on the lowest 10 moments was given in [\[Moch:2023tdj\]][crate::bib::Moch2023tdj].
///
/// These are approximations for fixed `nf` = 3, 4, 5 and 6 based on the
/// first 10 even moments together with small-x/large-x constraints.
/// The two sets providing the error estimate are called via `variation = 1`
/// and `variation = 2`.  Any other value of `variation` invokes their average.
pub fn gamma_gq(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let nf_ = nf as f64;
    let nf2 = nf_.pow(2);

    // Known large-x coefficients
    let x1L5cff = 1.3443073 * 10. - 5.4869684 * 0.1 * nf_;
    let x1L4cff = 3.7539831 * f64::pow(10., 2) - 3.4494742 * 10. * nf_ + 8.7791495 * 0.1 * nf2;
    let y1L5cff = 2.2222222 * 10. - 5.4869684 * 0.1 * nf_;
    let y1L4cff = 6.6242163 * f64::pow(10., 2) - 4.7992684 * 10. * nf_ + 8.7791495 * 0.1 * nf2;

    // Small-x, Casimir scaled from P_gg (approx. for bfkl1)
    let bfkl0 = -8.3086173 * f64::pow(10., 3) / 2.25;
    let bfkl1 = (-1.0691199 * f64::pow(10., 5) - nf_ * 9.9638304 * f64::pow(10., 2)) / 2.25;

    // Small-x double-logs with x^0
    let x0L6cff = 5.2235940 * 10. - 7.3744856 * nf_;
    let x0L5cff = -2.9221399 * f64::pow(10., 2) + 1.8436214 * nf_;
    let x0L4cff =
        7.3106077 * f64::pow(10., 3) - 3.7887135 * f64::pow(10., 2) * nf_ - 3.2438957 * 10. * nf2;

    // The resulting part of the function
    let P3GQ01 = bfkl0 * (-(6. / (-1. + n).powu(4)))
        + bfkl1 * 2. / (-1. + n).powu(3)
        + x0L6cff * 720. / n.powu(7)
        + x0L5cff * -120. / n.powu(6)
        + x0L4cff * 24. / n.powu(5)
        + x1L4cff * lm14(c)
        + x1L5cff * lm15(c)
        + y1L4cff * lm14m1(c)
        + y1L5cff * lm15m1(c);

    // The selected approximations for nf = 3, 4, 5, 6
    let P3gqApp1: Complex<f64>;
    let P3gqApp2: Complex<f64>;
    if nf == 3 {
        P3gqApp1 = P3GQ01 + 3.5 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 27891.0 * 1. / ((-1. + n) * n)
            - 309124.0 * 1. / n
            + 1056866.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 124735.0 * -1. / n.powu(2)
            - 16246.0 * 2. / n.powu(3)
            + 131175.0 * -6. / n.powu(4)
            + 4970.1 * lm13(c)
            + 60041.0 * lm12(c)
            + 343181.0 * lm11(c)
            - 958330.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3gqApp2 = P3GQ01 + 7.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 1139334.0 * 1. / ((-1. + n) * n)
            + 143008.0 * 1. / n
            - 290390.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 659492.0 * -1. / n.powu(2)
            + 303685.0 * 2. / n.powu(3)
            - 81867.0 * -6. / n.powu(4)
            + 1811.8 * lm13(c)
            - 465.9 * lm12(c)
            - 51206.0 * lm11(c)
            + 274249.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 4 {
        P3gqApp1 = P3GQ01 + 3.5 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 8302.8 * 1. / ((-1. + n) * n)
            - 347706.0 * 1. / n
            + 1105306.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 127650.0 * -1. / n.powu(2)
            - 29728.0 * 2. / n.powu(3)
            + 137537.0 * -6. / n.powu(4)
            + 4658.1 * lm13(c)
            + 59205.0 * lm12(c)
            + 345513.0 * lm11(c)
            - 995120.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3gqApp2 = P3GQ01 + 7.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 1129822.0 * 1. / ((-1. + n) * n)
            + 108527.0 * 1. / n
            - 254166.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 667254.0 * -1. / n.powu(2)
            + 293099.0 * 2. / n.powu(3)
            - 77437.0 * -6. / n.powu(4)
            + 1471.3 * lm13(c)
            - 1850.3 * lm12(c)
            - 52451.0 * lm11(c)
            + 248634.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 5 {
        P3gqApp1 =
            P3GQ01 + 3.5 * bfkl1 * (-(1. / (-1. + n).powu(2))) + 14035.0 * 1. / ((-1. + n) * n)
                - 384003.0 * 1. / n
                + 1152711.0 * (3. + n) / (2. + 3. * n + n.powu(2))
                - 126346.0 * -1. / n.powu(2)
                - 42967.0 * 2. / n.powu(3)
                + 144270.0 * -6. / n.powu(4)
                + 4385.5 * lm13(c)
                + 58688.0 * lm12(c)
                + 348988.0 * lm11(c)
                - 1031165.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3gqApp2 = P3GQ01 + 7.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 1117561.0 * 1. / ((-1. + n) * n)
            + 76329.0 * 1. / n
            - 218973.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 670799.0 * -1. / n.powu(2)
            + 282763.0 * 2. / n.powu(3)
            - 72633.0 * -6. / n.powu(4)
            + 1170.0 * lm13(c)
            - 2915.5 * lm12(c)
            - 52548.0 * lm11(c)
            + 223771.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 6 {
        P3gqApp1 =
            P3GQ01 + 3.5 * bfkl1 * (-(1. / (-1. + n).powu(2))) + 39203.0 * 1. / ((-1. + n) * n)
                - 417914.0 * 1. / n
                + 1199042.0 * (3. + n) / (2. + 3. * n + n.powu(2))
                - 120750.0 * -1. / n.powu(2)
                - 55941.0 * 2. / n.powu(3)
                + 151383.0 * -6. / n.powu(4)
                + 4149.2 * lm13(c)
                + 58466.0 * lm12(c)
                + 353589.0 * lm11(c)
                - 1066510.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3gqApp2 = P3GQ01 + 7.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 1102470.0 * 1. / ((-1. + n) * n)
            + 46517.0 * 1. / n
            - 184858.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            - 670056.0 * -1. / n.powu(2)
            + 272689.0 * 2. / n.powu(3)
            - 67453.0 * -6. / n.powu(4)
            + 905.0 * lm13(c)
            - 3686.2 * lm12(c)
            - 51523.0 * lm11(c)
            + 199594.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else {
        panic!("Select nf = 3..6 for N3LO evolution");
    }

    // We return (for now) one of the two error-band boundaries
    // or the present best estimate, their average
    let P3GQA = match variation {
        1 => P3gqApp1,
        2 => P3gqApp2,
        _ => 0.5 * (P3gqApp1 + P3gqApp2),
    };
    -P3GQA
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_reference_moments() {
        fn gq3_moment(N: usize, nf: f64) -> f64 {
            let nf2 = nf * nf;
            let nf3 = nf2 * nf;
            // From Eq. 9 of [\[Falcioni:2024xyt\]][crate::bib:Falcioni:2024xyt].
            let mom_list = [
                -16663.225488 + 4439.143749608238 * nf
                    - 202.55547919891168 * nf2
                    - 6.375390720235586 * nf3,
                -6565.7531450230645 + 1291.0067460871576 * nf
                    - 16.146190170051486 * nf2
                    - 0.8397634037808341 * nf3,
                -3937.479370556893 + 679.7185057363981 * nf
                    - 1.3720775271604673 * nf2
                    - 0.13979432728276966 * nf3,
                -2803.644107251366
                    + 436.39305738710254 * nf
                    + 1.8149462465491055 * nf2
                    + 0.07358858022119033 * nf3,
                -2179.48761 + 310.063163 * nf + 2.65636842 * nf2 + 0.15719522 * nf3,
                -1786.31231 + 234.383019 * nf + 2.82817592 * nf2 + 0.19211953 * nf3,
                -1516.59810 + 184.745296 * nf + 2.78076831 * nf2 + 0.20536518 * nf3,
                -1320.36106 + 150.076970 * nf + 2.66194730 * nf2 + 0.20798493 * nf3,
                -1171.29329 + 124.717778 * nf + 2.52563073 * nf2 + 0.20512226 * nf3,
                -1054.26140 + 105.497994 * nf + 2.39223358 * nf2 + 0.19938504 * nf3,
            ];
            mom_list[(N - 2) / 2]
        }
        for variation in [0, 1, 2] {
            for NF in [3, 4, 5, 6] {
                for N in [2.0, 4.0, 6.0, 8.0, 10., 12., 14., 16., 18., 20.] {
                    let mut c = Cache::new(cmplx!(N, 0.));
                    let test_value = gamma_gq(&mut c, NF, variation);
                    assert_approx_eq_cmplx!(
                        f64,
                        test_value,
                        cmplx!(gq3_moment(N as usize, NF as f64), 0.),
                        rel = if NF != 6 { 4e-4 } else { 2e-3 }
                    );
                }
            }
        }
    }
}
