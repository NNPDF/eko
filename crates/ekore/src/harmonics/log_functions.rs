/* Implementation of Mellin transformation of logarithms.

We provide transforms of:

- :math:`(1-x)\ln^k(1-x), \quad k = 1,2,3`
- :math:`\ln^k(1-x), \quad k = 1,3,4,5`
 */

use num::complex::Complex;
use num::pow;

/// Mellin transform of :math:`(1-x)\ln(1-x)`.
pub fn lm11m1(n: Complex<f64>, S1: Complex<f64>) -> Complex<f64> {
    1. / pow(1. + n, 2) - S1 / pow(1. + n, 2) - S1 / (n * pow(1. + n, 2))
}

/// Mellin transform of :math:`(1-x)\ln^2(1-x)`.
pub fn lm12m1(n: Complex<f64>, S1: Complex<f64>, S2: Complex<f64>) -> Complex<f64> {
    -2. / pow(1. + n, 3) - (2. * S1) / pow(1. + n, 2) + pow(S1, 2) / n - pow(S1, 2) / (1. + n)
        + S2 / n
        - S2 / (1. + n)
}

/// Mellin transform of :math:`(1-x)\ln^3(1-x)`.
pub fn lm13m1(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
) -> Complex<f64> {
    (3. * n * pow(1. + n, 2) * pow(S1, 2) - pow(1. + n, 3) * pow(S1, 3)
        + 3. * n * pow(1. + n, 2) * S2
        - 3. * (1. + n) * S1 * (-2. * n + pow(1. + n, 2) * S2)
        - 2. * (-3. * n + pow(1. + n, 3) * S3))
        / (n * pow(1. + n, 4))
}

/// Mellin transform of :math:`(1-x)\ln^4(1-x)`.
pub fn lm14m1(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
    S4: Complex<f64>,
) -> Complex<f64> {
    (-24. * n - 4. * n * pow(1. + n, 3) * pow(S1, 3) + pow(1. + n, 4) * pow(S1, 4)
        - 12. * n * pow(1. + n, 2) * S2
        + 3. * pow(1. + n, 4) * pow(S2, 2)
        + 6. * pow(1. + n, 2) * pow(S1, 2) * (-2. * n + pow(1. + n, 2) * S2)
        - 8. * n * S3
        - 24. * pow(n, 2) * S3
        - 24. * pow(n, 3) * S3
        - 8. * pow(n, 4) * S3
        - 4. * (1. + n)
            * S1
            * (3. * n * pow(1. + n, 2) * S2 - 2. * (-3. * n + pow(1. + n, 3) * S3))
        + 6. * S4
        + 24. * n * S4
        + 36. * pow(n, 2) * S4
        + 24. * pow(n, 3) * S4
        + 6. * pow(n, 4) * S4)
        / (n * pow(1. + n, 5))
}

/// Mellin transform of :math:`(1-x)\ln^5(1-x)`.
pub fn lm15m1(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
    S4: Complex<f64>,
    S5: Complex<f64>,
) -> Complex<f64> {
    (1. / (n * pow(1. + n, 6)))
        * (5. * n * pow(1. + n, 4) * pow(S1, 4) - pow(1. + n, 5) * pow(S1, 5)
            + 15. * n * pow(1. + n, 4) * pow(S2, 2)
            - 10. * pow(1. + n, 3) * pow(S1, 3) * (-2. * n + pow(1. + n, 2) * S2)
            + 40. * n * pow(1. + n, 3) * S3
            - 20. * pow(1. + n, 2) * S2 * (-3. * n + pow(1. + n, 3) * S3)
            + 10.
                * pow(S1, 2)
                * (3. * n * pow(1. + n, 4) * S2
                    - 2. * pow(1. + n, 2) * (-3. * n + pow(1. + n, 3) * S3))
            + 30. * n * pow(1. + n, 4) * S4
            - 5. * (1. + n)
                * S1
                * (-12. * n * pow(1. + n, 2) * S2 + 3. * pow(1. + n, 4) * pow(S2, 2)
                    - 8. * n * pow(1. + n, 3) * S3
                    + 6. * (-4. * n + pow(1. + n, 4) * S4))
            - 24. * (-5. * n + pow(1. + n, 5) * S5))
}

/// Mellin transform of :math:`\ln(1-x)`.
pub fn lm11(n: Complex<f64>, S1: Complex<f64>) -> Complex<f64> {
    -S1 / n
}

/// Mellin transform of :math:`\ln^2(1-x)`.
pub fn lm12(n: Complex<f64>, S1: Complex<f64>, S2: Complex<f64>) -> Complex<f64> {
    (pow(S1, 2) + S2) / n
}

/// Mellin transform of :math:`\ln^3(1-x)`.
pub fn lm13(n: Complex<f64>, S1: Complex<f64>, S2: Complex<f64>, S3: Complex<f64>) -> Complex<f64> {
    -(((pow(S1, 3) + ((3. * S1) * S2)) + (2. * S3)) / n)
}

/// Mellin transform of :math:`\ln^4(1-x)`.
pub fn lm14(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
    S4: Complex<f64>,
) -> Complex<f64> {
    ((((pow(S1, 4) + ((6. * pow(S1, 2)) * S2)) + (3. * pow(S2, 2))) + ((8. * S1) * S3)) + (6. * S4))
        / n
}

/// Mellin transform of :math:`\ln^5(1-x)`.
pub fn lm15(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
    S4: Complex<f64>,
    S5: Complex<f64>,
) -> Complex<f64> {
    -((((pow(S1, 5) + ((10. * pow(S1, 3)) * S2)) + ((20. * pow(S1, 2)) * S3))
        + ((15. * S1) * (pow(S2, 2) + (2. * S4))))
        + (4. * (((5. * S2) * S3) + (6. * S5))))
        / n
}

/// Mellin transform of :math:`(1-x)^2\ln(1-x)`.
pub fn lm11m2(n: Complex<f64>, S1: Complex<f64>) -> Complex<f64> {
    (5. + 3. * n - (2. * (1. + n) * (2. + n) * S1) / n) / (pow(pow(1. + n, 2) * (2. + n), 2))
}

/// Mellin transform of :math:`(1-x)^2\ln^2(1-x)`.
pub fn lm12m2(n: Complex<f64>, S1: Complex<f64>, S2: Complex<f64>) -> Complex<f64> {
    (2. * (n * (-9. - 8. * n + pow(n, 3))
        - n * (10. + 21. * n + 14. * pow(n, 2) + 3. * pow(n, 3)) * S1
        + pow(2. + 3. * n + pow(n, 2), 2) * pow(S1, 2)
        + pow(2. + 3. * n + pow(n, 2), 2) * S2))
        / pow(n * pow(1. + n, 3) * (2. + n), 3)
}

/// Mellin transform of :math:`(1-x)^2\ln^3(1-x)`.
pub fn lm13m2(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
) -> Complex<f64> {
    (-6. * n * (-17. - 21. * n - 2. * pow(n, 2) + 6. * pow(n, 3) + 2. * pow(n, 4))
        + 3. * n * (5. + 3. * n) * pow(2. + 3. * n + pow(n, 2), 2) * pow(S1, 2)
        - 2. * pow(2. + 3. * n + pow(n, 2), 3) * pow(S1, 3)
        + 3. * n * (5. + 3. * n) * pow(2. + 3. * n + pow(n, 2), 2) * S2
        - 6. * (2. + 3. * n + pow(n, 2))
            * S1
            * (n * (-9. - 8. * n + pow(n, 3)) + (2. + 3. * n + pow(pow(n, 2), 2) * S2)
                - 4. * pow(2. + 3. * n + pow(n, 2), 3) * S3))
        / (n * pow(pow(1. + n, 4) * (2. + n), 4))
}

/// Mellin transform of :math:`(1-x)^2\ln^4(1-x)`.
pub fn lm14m2(
    n: Complex<f64>,
    S1: Complex<f64>,
    S2: Complex<f64>,
    S3: Complex<f64>,
    S4: Complex<f64>,
) -> Complex<f64> {
    2. / (n * pow(1. + n, 5) * pow(2. + n, 5))
        * (12. * n * (-33. + n * (-54. + n * (-15. + n * (20. + 3. * n * (5. + n)))))
            - 2. * n * pow(1. + n, 3) * pow(2. + n, 3) * (5. + 3. * n) * pow(S1, 3)
            + pow(1. + n, 4) * pow(2. + n, 4) * pow(S1, 4)
            + 6. * n * pow(1. + n, 2) * pow(2. + n, 2) * (-9. - 8. * n + pow(n, 3)) * S2
            + 3. * pow(1. + n, 4) * pow(2. + n, 4) * pow(S2, 2)
            + 6. * pow(1. + n, 2)
                * pow(2. + n, 2)
                * pow(S1, 2)
                * (n * (-9. - 8. * n + pow(n, 3)) + pow(1. + n, 2) * pow(2. + n, 2) * S2)
            - 4. * n * pow(1. + n, 3) * pow(2. + n, 3) * (5. + 3. * n) * S3
            + 2. * (1. + n)
                * (2. + n)
                * S1
                * (6. * n * (-17. + n * (-21. + 2. * n * (-1. + n * (3. + n))))
                    - 3. * n * pow(1. + n, 2) * pow(2. + n, 2) * (5. + 3. * n) * S2
                    + 4. * pow(1. + n, 3) * pow(2. + n, 3) * S3)
            + 6. * pow(1. + n, 4) * pow(2. + n, 4) * S4)
}

#[cfg(test)]
mod tests {

    // use crate::harmonics as h;
    // use crate::{assert_approx_eq_cmplx};
    // use quad::integrate;
    // use num::complex::Complex;
    // use num::{pow};

    // #[test]
    // fn test_lm1pm2() {

    //     fn mellin_lm1pm1(x: f64, k: usize, n: f64) -> f64 {
    //         pow(pow(x, n - 1.0) * (1.0 - x) * (1.0 - x).ln(), k)
    //     }

    //     const Ns: [f64; 5] = [1.0, 1.5, 2.0, 2.34, 56.789];
    //     for N in Ns.iter().enumerate() {
    //         let n = Complex::new(*N.1, 0.);
    //         let S1 = h::w1::S1(n);
    //         let S2 = h::w2::S2(n);
    //         let S3 = h::w3::S3(n);
    //         let S4 = h::w4::S4(n);
    //         // let S4 = h::w5::S5(*N.1);

    //         let ref_values = vec![
    //                 h::log_functions::lm11m1(n, S1),
    //                 h::log_functions::lm12m1(n, S1, S2),
    //                 h::log_functions::lm13m1(n, S1, S2, S3),
    //                 h::log_functions::lm14m1(n, S1, S2, S3, S4),
    //                 h::log_functions::lm15m1(n, S1, S2, S3, S4, S4),
    //         ];

    //         for (k, &ref_value) in ref_values.iter().enumerate() {
    //                 let test_value = integrate(|x| mellin_lm1pm1(x, k as usize, n.re()), 0.0, 1.0, 1e-9).unwrap();
    //                 assert_approx_eq_cmplx!(f64, test_value, ref_value, epsilon = 1e-6);
    //             }
    //     }
    // }

    // def test_lm1pm1():
    // # test mellin transformation with some random N values
    // def mellin_lm1pm1(x, k, N):
    //     return x ** (N - 1) * (1 - x) * np.log(1 - x) ** k

    // Ns = [1.0, 1.5, 2.0, 2.34, 56.789]
    // for N in Ns:
    //     sx = hsx(N, 5)

    //     ref_values = {
    //         1: h.log_functions.lm11m1(N, S1),
    //         2: h.log_functions.lm12m1(N, S1, S2),
    //         3: h.log_functions.lm13m1(N, S1, S2, S3),
    //         4: h.log_functions.lm14m1(N, S1, S2, S3, S3),
    //         5: h.log_functions.lm15m1(N, S1, S2, S3, S3, sx[4]),
    //     }

    //     for k, ref in ref_values.items():
    //         test_value = quad(mellin_lm1pm1, 0, 1, args=(k, N))[0]
    //         np.testing.assert_allclose(test_value, ref)

    // def test_lm1p():
    // # test mellin transformation with some random N values
    // def mellin_lm1p(x, k, N):
    //     return x ** (N - 1) * np.log(1 - x) ** k

    // Ns = [1.0, 1.5, 2.0, 2.34, 56.789]
    // for N in Ns:
    //     sx = hsx(N, 5)

    //     ref_values = {
    //         1: h.log_functions.lm11(N, S1),
    //         2: h.log_functions.lm12(N, S1, S2),
    //         3: h.log_functions.lm13(N, S1, S2, S3),
    //         4: h.log_functions.lm14(N, S1, S2, S3, S3),
    //         5: h.log_functions.lm15(N, S1, S2, S3, S3, sx[4]),
    //     }

    //     for k in [1, 3, 4, 5]:
    //         test_value = quad(mellin_lm1p, 0, 1, args=(k, N))[0]
    //         np.testing.assert_allclose(test_value, ref_values[k])
}
