/// Compute the valence-like non-singlet anomalous dimension.
use num::complex::Complex;
use num::traits::Pow;

use super::gnsm;
use crate::harmonics::cache::{Cache, K};
use crate::harmonics::log_functions::{lm11m1, lm12m1, lm13m1};

/// Compute the sea-like non-singlet anomalous dimension.
fn gamma_nss(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);

    // nf^1: two approximations
    #[rustfmt::skip]
    let P3NSA11 = 2880. / n.powu(7)
        - 11672.4 / n.powu(6)
        + 12802.560000000001 / n.powu(5)
        - 7626.66 / n.powu(4)
        + 6593.2 / n.powu(3)
        - 3687.6 / n.powu(2)
        + 4989.2 / (1. + n)
        - 6596.93 / (2. + n)
        + 1607.73 / (3. + n)
        + 60.4 * lm12m1(n, S1, S2)
        + 4.685 * lm13m1(n, S1, S2, S3);
    #[rustfmt::skip]
    let P3NSA12 = -2880. / n.powu(7)
        + 4066.32 / n.powu(6)
        - 5682.24 / n.powu(5)
        + 5540.88 / n.powu(4)
        + 546.1 / n.powu(3)
        - 2987.83 / n.powu(2)
        + 2533.54 / n
        - 1502.75 / (1. + n)
        - 2297.56 / (2. + n)
        + 1266.77 / (3. + n)
        - 254.63 * lm11m1(n, S1)
        - 0.28953 * lm13m1(n, S1, S2, S3);

    // nf^2 (parametrized)
    #[rustfmt::skip]
    let P3NSSA2 = 47.4074 / n.powu(6)
        - 142.222 / n.powu(5)
        + 32.1201 / n.powu(4)
        - 132.824 / n.powu(3)
        + 647.397 / n.powu(2)
        + 19.7 * lm11m1(n, S1)
        - 3.43547 * lm12m1(n, S1, S2)
        - 1262.0951538579698 / n
        - 187.17000000000002 / (1. + n).powu(4)
        + 453.885 / (1. + n).powu(3)
        + 147.01749999999998 / (1. + n).powu(2)
        + 1614.1000000000001 / (1. + n)
        - 380.12500000000006 / (2. + n)
        - 42.575 / (3. + n)
        + (42.977500000000006 * S2) / n
        + (0.0900000000000047 * (477.52777777775293 + n) * S1) / (n.powu(2) * (1. + n));

    let P3NSSA: Complex<f64> = match variation {
        1 => (nf as f64) * P3NSA11 + (nf as f64).pow(2) * P3NSSA2,
        2 => (nf as f64) * P3NSA12 + (nf as f64).pow(2) * P3NSSA2,
        _ => 0.5 * (nf as f64) * (P3NSA11 + P3NSA12) + (nf as f64).pow(2) * P3NSSA2,
    };
    -P3NSSA
}

/// Compute the valence non-singlet anomalous dimension.
///
/// See [gamma_nsm][super::gamma_nsm] for implementation details.
pub fn gamma_nsv(c: &mut Cache, nf: u8, variation: u8) -> Complex<f64> {
    gnsm::gamma_nsm(c, nf, variation) + gamma_nss(c, nf, variation)
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
        let refs: [f64; 3] = [-0.01100459, -0. - 0.00779938, -0.0142098];
        for var in [0, 1, 2] {
            let test_value = gamma_nss(&mut c, NF, var);
            assert_approx_eq_cmplx!(f64, test_value, cmplx!(refs[var as usize], 0.), rel = 5e-6);
        }
    }

    #[test]
    fn test_reference_moments() {
        let NF = 4;
        let nss_nf4_refs: [f64; 8] = [
            50.10532524,
            39.001939964,
            21.141505811200002,
            12.4834195012,
            8.0006134908,
            5.4610639744,
            3.9114290952,
            2.90857799,
        ];
        for N in [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0] {
            let mut c = Cache::new(cmplx!(N, 0.));
            let test_value = gamma_nss(&mut c, NF, 0);
            assert_approx_eq_cmplx!(
                f64,
                test_value,
                cmplx!(nss_nf4_refs[((N - 2.) / 2.) as usize], 0.),
                rel = 5e-5
            );
        }
    }
}
