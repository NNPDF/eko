//! Mellin transformation of logarithms.
//!
//! We provide transforms of:
//!
//!  * $\ln^k(1-x), \quad k = 1,2,3,4$
//!  * $(1-x)\ln^k(1-x), \quad k = 1,2,3$
//!  * $(1-x)^2\ln^k(1-x), \quad k = 1,2,3$
use super::cache::{Cache, K};
use num::complex::Complex;
use num::pow;

/// Mellin transform of $(1-x)\ln(1-x)$.
pub fn lm11m1(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    1. / (1. + n).powu(2) - S1 / (1. + n).powu(2) - S1 / (n * (1. + n).powu(2))
}

/// Mellin transform of $(1-x)\ln^2(1-x)$.
pub fn lm12m1(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    -2. / (1. + n).powu(3) - (2. * S1) / (1. + n).powu(2) + S1.powu(2) / n - S1.powu(2) / (1. + n)
        + S2 / n
        - S2 / (1. + n)
}

/// Mellin transform of $(1-x)\ln^3(1-x)$.
pub fn lm13m1(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    (3. * n * (1. + n).powu(2) * S1.powu(2) - (1. + n).powu(3) * S1.powu(3)
        + 3. * n * (1. + n).powu(2) * S2
        - 3. * (1. + n) * S1 * (-2. * n + (1. + n).powu(2) * S2)
        - 2. * (-3. * n + (1. + n).powu(3) * S3))
        / (n * (1. + n).powu(4))
}

/// Mellin transform of $(1-x)\ln^4(1-x)$.
pub fn lm14m1(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    (-24. * n - 4. * n * (1. + n).powu(3) * S1.powu(3) + (1. + n).powu(4) * S1.powu(4)
        - 12. * n * (1. + n).powu(2) * S2
        + 3. * (1. + n).powu(4) * S2.powu(2)
        + 6. * (1. + n).powu(2) * S1.powu(2) * (-2. * n + (1. + n).powu(2) * S2)
        - 8. * n * S3
        - 24. * n.powu(2) * S3
        - 24. * n.powu(3) * S3
        - 8. * n.powu(4) * S3
        - 4. * (1. + n)
            * S1
            * (3. * n * (1. + n).powu(2) * S2 - 2. * (-3. * n + (1. + n).powu(3) * S3))
        + 6. * S4
        + 24. * n * S4
        + 36. * n.powu(2) * S4
        + 24. * n.powu(3) * S4
        + 6. * n.powu(4) * S4)
        / (n * (1. + n).powu(5))
}

/// Mellin transform of $(1-x)\ln^5(1-x)$.
pub fn lm15m1(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    let S5 = c.get(K::S5);
    (1. / (n * (1. + n).powu(6)))
        * (5. * n * (1. + n).powu(4) * S1.powu(4) - (1. + n).powu(5) * S1.powu(5)
            + 15. * n * (1. + n).powu(4) * S2.powu(2)
            - 10. * (1. + n).powu(3) * S1.powu(3) * (-2. * n + (1. + n).powu(2) * S2)
            + 40. * n * (1. + n).powu(3) * S3
            - 20. * (1. + n).powu(2) * S2 * (-3. * n + (1. + n).powu(3) * S3)
            + 10.
                * S1.powu(2)
                * (3. * n * (1. + n).powu(4) * S2
                    - 2. * (1. + n).powu(2) * (-3. * n + (1. + n).powu(3) * S3))
            + 30. * n * (1. + n).powu(4) * S4
            - 5. * (1. + n)
                * S1
                * (-12. * n * (1. + n).powu(2) * S2 + 3. * (1. + n).powu(4) * S2.powu(2)
                    - 8. * n * (1. + n).powu(3) * S3
                    + 6. * (-4. * n + (1. + n).powu(4) * S4))
            - 24. * (-5. * n + (1. + n).powu(5) * S5))
}

/// Mellin transform of $\ln(1-x)$.
pub fn lm11(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    -S1 / n
}

/// Mellin transform of $\ln^2(1-x)$.
pub fn lm12(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    (S1.powu(2) + S2) / n
}

/// Mellin transform of $\ln^3(1-x)$.
pub fn lm13(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    -((S1.powu(3) + (3. * S1) * S2 + 2. * S3) / n)
}

/// Mellin transform of $\ln^4(1-x)$.
pub fn lm14(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    (S1.powu(4) + 6. * S1.powu(2) * S2 + 3. * S2.powu(2) + 8. * S1 * S3 + 6. * S4) / n
}

/// Mellin transform of $\ln^5(1-x)$.
pub fn lm15(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    let S5 = c.get(K::S5);
    -(S1.powu(5)
        + 10. * S1.powu(3) * S2
        + 20. * S1.powu(2) * S3
        + 15. * S1 * (S2.powu(2) + 2. * S4)
        + 4. * (5. * S2 * S3 + 6. * S5))
        / n
}

/// Mellin transform of $(1-x)^2\ln(1-x)$.
pub fn lm11m2(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    (5. + 3. * n - (2. * (1. + n) * (2. + n) * S1) / n) / ((1. + n).powu(2) * (2. + n).powu(2))
}

/// Mellin transform of $(1-x)^2\ln^2(1-x)$.
pub fn lm12m2(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    (2. * (n * (-9. - 8. * n + n.powu(3))
        - n * (10. + 21. * n + 14. * n.powu(2) + 3. * n.powu(3)) * S1
        + pow(2. + 3. * n + n.powu(2), 2) * S1.powu(2)
        + pow(2. + 3. * n + n.powu(2), 2) * S2))
        / (n * (1. + n).powu(3) * (2. + n).powu(3))
}

/// Mellin transform of $(1-x)^2\ln^3(1-x)$.
pub fn lm13m2(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    (-6. * n * (-17. - 21. * n - 2. * n.powu(2) + 6. * n.powu(3) + 2. * n.powu(4))
        + 3. * n * (5. + 3. * n) * pow(2. + 3. * n + n.powu(2), 2) * S1.powu(2)
        - 2. * pow(2. + 3. * n + n.powu(2), 3) * S1.powu(3)
        + 3. * n * (5. + 3. * n) * pow(2. + 3. * n + n.powu(2), 2) * S2
        - 6. * (2. + 3. * n + n.powu(2))
            * S1
            * (n * (-9. - 8. * n + n.powu(3)) + pow(2. + 3. * n + n.powu(2), 2) * S2)
        - 4. * pow(2. + 3. * n + n.powu(2), 3) * S3)
        / (n * (1. + n).powu(4) * (2. + n).powu(4))
}

/// Mellin transform of $(1-x)^2\ln^4(1-x)$.
pub fn lm14m2(c: &mut Cache) -> Complex<f64> {
    let n = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S4 = c.get(K::S4);
    2. / (n * (1. + n).powu(5) * pow(2. + n, 5))
        * (12. * n * (-33. + n * (-54. + n * (-15. + n * (20. + 3. * n * (5. + n)))))
            - 2. * n * (1. + n).powu(3) * pow(2. + n, 3) * (5. + 3. * n) * S1.powu(3)
            + (1. + n).powu(4) * pow(2. + n, 4) * S1.powu(4)
            + 6. * n * (1. + n).powu(2) * pow(2. + n, 2) * (-9. - 8. * n + n.powu(3)) * S2
            + 3. * (1. + n).powu(4) * pow(2. + n, 4) * S2.powu(2)
            + 6. * (1. + n).powu(2)
                * pow(2. + n, 2)
                * S1.powu(2)
                * (n * (-9. - 8. * n + n.powu(3)) + (1. + n).powu(2) * pow(2. + n, 2) * S2)
            - 4. * n * (1. + n).powu(3) * pow(2. + n, 3) * (5. + 3. * n) * S3
            + 2. * (1. + n)
                * (2. + n)
                * S1
                * (6. * n * (-17. + n * (-21. + 2. * n * (-1. + n * (3. + n))))
                    - 3. * n * (1. + n).powu(2) * pow(2. + n, 2) * (5. + 3. * n) * S2
                    + 4. * (1. + n).powu(3) * pow(2. + n, 3) * S3)
            + 6. * (1. + n).powu(4) * pow(2. + n, 4) * S4)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::traits::Pow;
    use rgsl::integration::qng;

    #[test]
    fn test_lm1Xm1() {
        fn mellin_lm1Xm1(x: f64, k: f64, n: f64) -> f64 {
            f64::pow(x, n - 1.0) * (1.0 - x) * f64::pow((1.0 - x).ln(), k)
        }

        const NS: [f64; 5] = [1.0, 1.5, 2.0, 2.34, 56.789];
        for N in NS {
            let mut c = Cache::new(cmplx!(N, 0.));

            let ref_values = [
                lm11m1(&mut c),
                lm12m1(&mut c),
                lm13m1(&mut c),
                lm14m1(&mut c),
            ];

            for (k, &ref_value) in ref_values.iter().enumerate() {
                let test_value = qng(
                    |x| mellin_lm1Xm1(x, k as f64 + 1.0, N),
                    0.0,
                    1.0,
                    1e-3,
                    1e-4,
                )
                .unwrap();
                assert_approx_eq_cmplx!(f64, cmplx!(test_value.0, 0.0), ref_value, epsilon = 1e-6);
            }

            // here the integration is poorly convergent
            let ref_values5 = lm15m1(&mut c);
            let test_value =
                qng(|x| mellin_lm1Xm1(x, 4.0 + 1.0, N), 0.0, 0.9999, 1e-2, 1e-4).unwrap();
            assert_approx_eq_cmplx!(f64, cmplx!(test_value.0, 0.0), ref_values5, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_lmXm2() {
        fn mellin_lm1Xm2(x: f64, k: f64, n: f64) -> f64 {
            f64::pow(x, n - 1.0) * f64::pow(1.0 - x, 2) * f64::pow((1.0 - x).ln(), k)
        }

        const NS: [f64; 5] = [1.0, 1.5, 2.0, 2.34, 56.789];
        for N in NS {
            let mut c = Cache::new(cmplx!(N, 0.));

            let ref_values = [
                lm11m2(&mut c),
                lm12m2(&mut c),
                lm13m2(&mut c),
                lm14m2(&mut c),
            ];

            for (k, &ref_value) in ref_values.iter().enumerate() {
                let test_value = qng(
                    |x| mellin_lm1Xm2(x, k as f64 + 1.0, N),
                    0.0,
                    1.0,
                    1e-4,
                    1e-4,
                )
                .unwrap();
                assert_approx_eq_cmplx!(f64, cmplx!(test_value.0, 0.0), ref_value, epsilon = 1e-6);
            }
        }
    }
}
