/// Compute the singlet gluon-to-quark anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::constants::ZETA2;
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11, lm12, lm13, lm14, lm14m1, lm15, lm15m1};

/// Compute the singlet gluon-to-quark anomalous dimension.
///
/// The routine is taken from [\[Falcioni:2023vqq\]][crate::bib::Falcioni2023vqq],
/// with the update for :math:`n_f=6` from [\[Falcioni:2025hfz\]][crate::bib::Falcioni:2025hfz].
///
/// These are approximations for fixed `nf` = 3, 4, 5 and 6 based on the
/// first 10 even moments together with small-x/large-x constraints.
/// The two sets indicating the error estimate are called via `variation = 1`
/// and `variation = 2`.  Any other value of `variation` invokes their average.
pub fn gamma_qg(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let nf_ = nf as f64;
    let nf2 = nf_.pow(2);
    let nf3 = nf_.pow(3);

    // Known large-x coefficients
    let x1L5cff = 1.8518519 * nf_ - 4.1152263 * 0.1 * nf2;
    let x1L4cff = 3.5687794 * 10. * nf_ - 3.5116598 * nf2 - 8.2304527 * 0.01 * nf3;
    let y1L5cff = 2.8806584 * nf_ + 8.2304527 * 0.1 * nf2;
    let y1L4cff = -4.0511391 * 10. * nf_ + 5.5418381 * nf2 + 1.6460905 * 0.1 * nf3;

    // Known small-x coefficients
    let bfkl1 = 3.9357613 * f64::pow(10., 3) * nf_;
    let x0L6cff = -1.9588477 * 10. * nf_ + 2.7654321 * nf2;
    let x0L5cff = 2.1573663 * 10. * nf_ + 1.7244444 * 10. * nf2;
    let x0L4cff =
        -2.8667643 * f64::pow(10., 3) * nf_ + 3.0122403 * f64::pow(10., 2) * nf2 + 4.1316872 * nf3;

    // The resulting part of the function
    let P3QG01 = bfkl1 * 2. / (-1. + n).powu(3)
        + x0L6cff * 720. / n.powu(7)
        + x0L5cff * -120. / n.powu(6)
        + x0L4cff * 24. / n.powu(5)
        + x1L4cff * lm14(c)
        + x1L5cff * lm15(c)
        + y1L4cff * lm14m1(c)
        + y1L5cff * lm15m1(c);

    // The selected approximations for nf = 3, 4, 5, 6
    let P3qgApp1: Complex<f64>;
    let P3qgApp2: Complex<f64>;
    if nf == 3 {
        P3qgApp1 = P3QG01 + 187500.0 * -(1. / (-1. + n).powu(2)) + 826060.0 * 1. / ((-1. + n) * n)
            - 150474.0 * 1. / n
            + 226254.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 577733.0 * -1. / n.powu(2)
            - 180747.0 * 2. / n.powu(3)
            + 95411.0 * -6. / n.powu(4)
            + 119.8 * lm13(c)
            + 7156.3 * lm12(c)
            + 45790.0 * lm11(c)
            - 95682.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3qgApp2 = P3QG01 + 135000.0 * -(1. / (-1. + n).powu(2)) + 484742.0 * 1. / ((-1. + n) * n)
            - 11627.0 * 1. / n
            - 187478.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 413512.0 * -1. / n.powu(2)
            - 82500.0 * 2. / n.powu(3)
            + 29987.0 * -6. / n.powu(4)
            - 850.1 * lm13(c)
            - 11425.0 * lm12(c)
            - 75323.0 * lm11(c)
            + 282836.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 4 {
        P3qgApp1 = P3QG01 + 250000.0 * -(1. / (-1. + n).powu(2)) + 1089180.0 * 1. / ((-1. + n) * n)
            - 241088.0 * 1. / n
            + 342902.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 720081.0 * -1. / n.powu(2)
            - 247071.0 * 2. / n.powu(3)
            + 126405.0 * -6. / n.powu(4)
            + 272.4 * lm13(c)
            + 10911.0 * lm12(c)
            + 60563.0 * lm11(c)
            - 161448.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3qgApp2 = P3QG01 + 180000.0 * -(1. / (-1. + n).powu(2)) + 634090.0 * 1. / ((-1. + n) * n)
            - 55958.0 * 1. / n
            - 208744.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 501120.0 * -1. / n.powu(2)
            - 116073.0 * 2. / n.powu(3)
            + 39173.0 * -6. / n.powu(4)
            - 1020.8 * lm13(c)
            - 13864.0 * lm12(c)
            - 100922.0 * lm11(c)
            + 343243.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 5 {
        P3qgApp1 = P3QG01 + 312500.0 * -(1. / (-1. + n).powu(2)) + 1345700.0 * 1. / ((-1. + n) * n)
            - 350466.0 * 1. / n
            + 480028.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 837903.0 * -1. / n.powu(2)
            - 315928.0 * 2. / n.powu(3)
            + 157086.0 * -6. / n.powu(4)
            + 472.7 * lm13(c)
            + 15415.0 * lm12(c)
            + 75644.0 * lm11(c)
            - 244869.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3qgApp2 = P3QG01 + 225000.0 * -(1. / (-1. + n).powu(2)) + 776837.0 * 1. / ((-1. + n) * n)
            - 119054.0 * 1. / n
            - 209530.0 * (3. + n) / (2. + 3. * n + n.powu(2))
            + 564202.0 * -1. / n.powu(2)
            - 152181.0 * 2. / n.powu(3)
            + 48046.0 * -6. / n.powu(4)
            - 1143.8 * lm13(c)
            - 15553.0 * lm12(c)
            - 126212.0 * lm11(c)
            + 385995.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 6 {
        P3qgApp1 =
            P3QG01 + 375000.0 * -(1. / (-1. + n).powu(2)) + 1595330.0 * 1. / ((-1. + n) * n)
                - 477729.0 * 1. / n
                + 637552.0 * (3. + n) / (2. + 3. * n + n.powu(2))
                + 931556.0 * -1. / n.powu(2)
                - 387017.0 * 2. / n.powu(3)
                + 187509.0 * -6. / n.powu(4)
                + 715.5 * lm13(c)
                + 20710.0 * lm12(c)
                + 91373.0 * lm11(c)
                - 346374.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
        P3qgApp2 =
            P3QG01 + 270000.0 * -(1. / (-1. + n).powu(2)) + 912695.0 * 1. / ((-1. + n) * n)
                - 200034.0 * 1. / n
                - 189918.0 * (3. + n) / (2. + 3. * n + n.powu(2))
                + 603114.0 * -1. / n.powu(2)
                - 190521.0 * 2. / n.powu(3)
                + 56661.0 * -6. / n.powu(4)
                - 1224.3 * lm13(c)
                - 16453.0 * lm12(c)
                - 150856.0 * lm11(c)
                + 410661.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else {
        panic!("Select nf=3,..,6 for N3LO evolution");
    }

    // We return (for now) one of the two error-band boundaries
    // or the present best estimate, their average
    let P3QGA = match variation {
        1 => P3qgApp1,
        2 => P3qgApp2,
        _ => 0.5 * (P3qgApp1 + P3qgApp2),
    };
    -P3QGA
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_reference_moments() {
        fn qg3_moment(N: usize, nf: f64) -> f64 {
            let nf2 = nf * nf;
            let nf3 = nf2 * nf;
            // From Eq. 4 of [Falcioni:2023vqq].
            let mom_list = [
                -654.4627782205557 * nf + 245.61061978871788 * nf2 - 0.9249909688301847 * nf3,
                290.31106867034487 * nf - 76.51672403736478 * nf2 - 4.911625629947491 * nf3,
                335.80080466045274 * nf - 124.57102255718002 * nf2 - 4.193871425027802 * nf3,
                294.58768309440677 * nf - 135.3767647714609 * nf2 - 3.609775642729055 * nf3,
                241.6153399044715 * nf - 135.18742470907011 * nf2 - 3.189394834180898 * nf3,
                191.97124640777176 * nf - 131.16316638326697 * nf2 - 2.8771044305171913 * nf3,
                148.5682948286098 * nf - 125.82310814280595 * nf2 - 2.635918561148907 * nf3,
                111.34042526856348 * nf - 120.16819876888667 * nf2 - 2.4433790398202664 * nf3,
                79.51561588665083 * nf - 114.61713540075442 * nf2 - 2.28548686108789 * nf3,
                52.24329555231736 * nf - 109.34248910828198 * nf2 - 2.1531537251387527 * nf3,
            ];
            mom_list[(N - 2) / 2]
        }
        for variation in [0, 1, 2] {
            for NF in [3, 4, 5, 6] {
                for N in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0] {
                    let mut c = Cache::new(cmplx!(N, 0.));
                    let test_value = gamma_qg(&mut c, NF, variation);
                    assert_approx_eq_cmplx!(
                        f64,
                        test_value,
                        cmplx!(qg3_moment(N as usize, NF as f64), 0.),
                        rel = 8e-4
                    );
                }
            }
        }
    }
}
