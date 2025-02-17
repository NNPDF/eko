/// Compute the singlet quark-to-gluon anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use crate::constants::ZETA2;
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11, lm12, lm12m1, lm13, lm14, lm14m1, lm15, lm15m1};

// The routine is taken from [\[Falcioni:2024xyt\]][crate::bib:Falcioni:2024xyt].
pub fn gamma_gq(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    let S5 = c.get(K::S5);
    let nf2 = nf.pow(2) as f64;

    //Known large-x coefficients
    let x1L5cff = 1.3443073 * 10. - 5.4869684 * 0.1 * nf as f64;
    let x1L4cff =
        3.7539831 * f64::pow(10., 2) - 3.4494742 * 10. * nf as f64 + 8.7791495 * 0.1 * nf2;
    let y1L5cff = 2.2222222 * 10. - 5.4869684 * 0.1 * nf as f64;
    let y1L4cff =
        6.6242163 * f64::pow(10., 2) - 4.7992684 * 10. * nf as f64 + 8.7791495 * 0.1 * nf2;

    //Small-x, Casimir scaled from P_gg (approx. for bfkl1)
    let bfkl0 = -8.3086173 * f64::pow(10., 3) / 2.25;
    let bfkl1 = (-1.0691199 * f64::pow(10., 5) - nf as f64 * 9.9638304 * f64::pow(10., 2)) / 2.25;

    //Small-x double-logs with x^0
    let x0L6cff = 5.2235940 * 10. - 7.3744856 * nf as f64;
    let x0L5cff = -2.9221399 * f64::pow(10., 2) + 1.8436214 * nf as f64;
    let x0L4cff = 7.3106077 * f64::pow(10., 3)
        - 3.7887135 * f64::pow(10., 2) * nf as f64
        - 3.2438957 * 10. * nf2;

    //The resulting part of the function
    let P3GQ01 = bfkl0 * (-(6. / (-1. + n).powu(4)))
        + bfkl1 * 2. / (-1. + n).powu(3)
        + x0L6cff * 720. / n.powu(7)
        + x0L5cff * -120. / n.powu(6)
        + x0L4cff * 24. / n.powu(5)
        + x1L4cff * lm14(n, S1, S2, S3, S4)
        + x1L5cff * lm15(n, S1, S2, S3, S4, S5)
        + y1L4cff * lm14m1(n, S1, S2, S3, S4)
        + y1L5cff * lm15m1(n, S1, S2, S3, S4, S5);

    //The selected approximations for nf = 3, 4, 5
    let P3gqApp1;
    let P3gqApp2;
    if nf == 3 {
        P3gqApp1 = P3GQ01 + 6.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 744384.0 * 1. / ((-1. + n) * n)
            + 2453640.0 * 1. / n
            - 1540404.0 * (2. / (1. + n) + 1. / (2. + n))
            + 1933026.0 * -1. / n.powu(2)
            + 1142069.0 * 2. / n.powu(3)
            + 162196.0 * -6. / n.powu(4)
            - 2172.1 * lm13(n, S1, S2, S3)
            - 93264.1 * lm12(n, S1, S2)
            - 786973.0 * lm11(n, S1)
            + 875383.0 * lm12m1(n, S1, S2);
        P3gqApp2 =
            P3GQ01 + 3.0 * bfkl1 * (-(1. / (-1. + n).powu(2))) + 142414.0 * 1. / ((-1. + n) * n)
                - 326525.0 * 1. / n
                + 2159787.0 * ((3. + n) / (2. + 3. * n + n.powu(2)))
                - 289064.0 * -1. / n.powu(2)
                - 176358.0 * 2. / n.powu(3)
                + 156541.0 * -6. / n.powu(4)
                + 9016.5 * lm13(n, S1, S2, S3)
                + 136063.0 * lm12(n, S1, S2)
                + 829482.0 * lm11(n, S1)
                - 2359050.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 4 {
        P3gqApp1 = P3GQ01 + 6.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 743535.0 * 1. / ((-1. + n) * n)
            + 2125286.0 * 1. / n
            - 1332472.0 * (2. / (1. + n) + 1. / (2. + n))
            + 1631173.0 * -1. / n.powu(2)
            + 1015255.0 * 2. / n.powu(3)
            + 142612.0 * -6. / n.powu(4)
            - 1910.4 * lm13(n, S1, S2, S3)
            - 80851.0 * lm12(n, S1, S2)
            - 680219.0 * lm11(n, S1)
            + 752733.0 * lm12m1(n, S1, S2);
        P3gqApp2 =
            P3GQ01 + 3.0 * bfkl1 * (-(1. / (-1. + n).powu(2))) + 160568.0 * 1. / ((-1. + n) * n)
                - 361207.0 * 1. / n
                + 2048948.0 * ((3. + n) / (2. + 3. * n + n.powu(2)))
                - 245963.0 * -1. / n.powu(2)
                - 171312.0 * 2. / n.powu(3)
                + 163099.0 * -6. / n.powu(4)
                + 8132.2 * lm13(n, S1, S2, S3)
                + 124425.0 * lm12(n, S1, S2)
                + 762435.0 * lm11(n, S1)
                - 2193335.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else if nf == 5 {
        P3gqApp1 = P3GQ01 + 6.0 * bfkl1 * (-(1. / (-1. + n).powu(2)))
            - 785864.0 * 1. / ((-1. + n) * n)
            + 285034.0 * 1. / n
            - 131648.0 * (2. / (1. + n) + 1. / (2. + n))
            - 162840.0 * -1. / n.powu(2)
            + 321220.0 * 2. / n.powu(3)
            + 12688.0 * -6. / n.powu(4)
            + 1423.4 * lm13(n, S1, S2, S3)
            + 1278.9 * lm12(n, S1, S2)
            - 30919.9 * lm11(n, S1)
            + 47588.0 * lm12m1(n, S1, S2);
        P3gqApp2 =
            P3GQ01 + 3.0 * bfkl1 * (-(1. / (-1. + n).powu(2))) + 177094.0 * 1. / ((-1. + n) * n)
                - 470694.0 * 1. / n
                + 1348823.0 * ((3. + n) / (2. + 3. * n + n.powu(2)))
                - 52985.0 * -1. / n.powu(2)
                - 87354.0 * 2. / n.powu(3)
                + 176885.0 * -6. / n.powu(4)
                + 4748.8 * lm13(n, S1, S2, S3)
                + 65811.9 * lm12(n, S1, S2)
                + 396390.0 * lm11(n, S1)
                - 1190212.0 * (S1 - n * (ZETA2 - S2)) / n.powu(2);
    } else {
        panic!("nf=6 is not available at N3LO");
    }

    // We return (for now) one of the two error-band boundaries
    // or the present best estimate, their average
    let P3GQA = Complex::from(match variation {
        1 => P3gqApp1,
        2 => P3gqApp2,
        _ => 0.5 * (P3gqApp1 + P3gqApp2),
    });
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
            ];
            mom_list[(N - 2) / 2]
        }
        for variation in [0, 1, 2] {
            for NF in [3, 4, 5] {
                for N in [2.0, 4.0, 6.0, 8.0] {
                    let mut c = Cache::new(cmplx!(N, 0.));
                    let test_value = gamma_gq(&mut c, NF, variation);
                    assert_approx_eq_cmplx!(
                        f64,
                        test_value,
                        cmplx!(gq3_moment(N as usize, NF as f64), 0.),
                        rel = 4e-4
                    );
                }
            }
        }
    }
}
