//! |NNLO| |QCD|

use num::complex::Complex;
use num::traits::Pow;
use num::Zero;

use crate::cmplx;
use crate::constants::{CF, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};

use crate::operator_matrix_elements::unpolarized::spacelike::as1;

/// |NNLO| light-light non-singlet |OME|.
/// It is given in Eq.() of
pub fn A_qq_ns(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S3 = c.get(K::S3);
    let S1m = c.get(K::S1) - 1. / N;
    let S2m = c.get(K::S2) - 1. / N.powu(2);

    let a_qq_l0 = -224.0 / 27.0 * (S1 - 1.0 / N) - 8.0 / 3.0 * ZETA3
        + 40. / 9.0 * ZETA2
        + 73.0 / 18.0
        + 44.0 / 27.0 / N
        - 268.0 / 27.0 / (N + 1.0)
        + 8.0 / 3.0 * (-1.0 / N.powu(2) + 1.0 / (N + 1.0).powu(2))
        + 20.0 / 9.0 * (S2 - 1.0 / N.powu(2) - ZETA2 + S2 + 1.0 / (N + 1.0).powu(2) - ZETA2)
        + 2.0 / 3.0
            * (-2.0 * (S3 - 1.0 / N.powu(3) - ZETA3)
                - 2.0 * (S3 + 1.0 / (N + 1.0).powu(3) - ZETA3));

    let a_qq_l1 = 2. * (-12. - 28. * N + 9. * N.powu(2) + 34. * N.powu(3) - 3. * N.powu(4))
        / (9. * (N * (N + 1.)).powu(2))
        + 80. / 9. * S1m
        - 16. / 3. * S2m;

    let a_qq_l2 = -2. * ((2. + N - 3. * N.powu(2)) / (3. * N * (N + 1.)) + 4. / 3. * S1m);

    CF * TR * (a_qq_l2 * L.pow(2) + a_qq_l1 * L + a_qq_l0)
}

/// |NNLO| heavy-light pure-singlet |OME|
/// It is given in Eq.() of
pub fn A_hq_ps(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n();
    let S2 = c.get(K::S1);
    let F1M = 1.0 / (N - 1.0) * (ZETA2 - (S2 - 1.0 / N.powu(2)));
    let F11 = 1.0 / (N + 1.0) * (ZETA2 - (S2 + 1.0 / (N + 1.0).powu(2)));
    let F12 = 1.0 / (N + 2.0) * (ZETA2 - (S2 + 1.0 / (N + 1.0).powu(2) + 1.0 / (N + 2.0).powu(2)));
    let F21 = -F11 / (N + 1.0);

    let a_hq_l0 = -(32.0 / 3.0 / (N - 1.0) + 8.0 * (1.0 / N - 1.0 / (N + 1.0))
        - 32.0 / 3.0 * 1.0 / (N + 2.0))
        * ZETA2
        - 448.0 / 27.0 / (N - 1.0)
        - 4.0 / 3.0 / N
        - 124.0 / 3.0 * 1.0 / (N + 1.0)
        + 1600.0 / 27.0 / (N + 2.0)
        - 4.0 / 3.0 * (-6.0 / N.powu(4) - 6.0 / (N + 1.0).powu(4))
        + 2.0 * 2.0 / N.powu(3)
        + 10.0 * 2.0 / (N + 1.0).powu(3)
        + 16.0 / 3.0 * 2.0 / (N + 2.0).powu(3)
        - 16.0 * ZETA2 * (-1.0 / N.powu(2) - 1.0 / (N + 1.0).powu(2))
        + 56.0 / 3.0 / N.powu(2)
        + 88.0 / 3.0 / (N + 1.0).powu(2)
        + 448.0 / 9.0 / (N + 2.0).powu(2)
        + 32.0 / 3.0 * F1M
        + 8.0 * ((ZETA2 - S2) / N - F11)
        - 32.0 / 3.0 * F12
        + 16.0 * (-(ZETA2 - S2) / N.powu(2) + F21);

    let a_hq_l1 = 8. * (2. + N * (5. + N)) * (4. + N * (4. + N * (7. + 5. * N)))
        / ((N - 1.) * (N + 2.).powu(2) * (N * (N + 1.)).powu(3));

    let a_hq_l2 =
        -4. * (2. + N + N.powu(2)).powu(2) / ((N - 1.) * (N + 2.) * (N * (N + 1.)).powu(2));

    CF * TR * (a_hq_l2 * L.pow(2) + a_hq_l1 * L + a_hq_l0)
}

/// |NNLO| heavy-gluon |OME|
/// It is given in Eq.() of
pub fn A_hg(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let Sm2e = c.get(K::Sm2e);
    let S3 = c.get(K::S3);
    let Sm21e = c.get(K::Sm21e);
    let Sm3e = c.get(K::Sm3e);
    let S1m = S1 - 1. / N;
    let S2m = S2 - 1. / N.powu(2);

    #[rustfmt::skip]
    let a_hg_l0 =
        -(
            3084.
            + 192. / N.powu(4)
            + 1056. / N.powu(3)
            + 2496. / N.powu(2)
            + 2928. / N
            + 2970. * N
            + 1782. * N.powu(2)
            + 6. * N.powu(3)
            - 1194. * N.powu(4)
            - 1152. * N.powu(5)
            - 516. * N.powu(6)
            - 120. * N.powu(7)
            - 12. * N.powu(8)
        )
        / ((N - 1.) * ((1. + N) * (2. + N)).powu(4))
        + (
            764.
            - 16. / N.powu(4)
            - 80. / N.powu(3)
            - 100. / N.powu(2)
            + 3. * 72. / N
            + 208. * N.powu(3)
            + 3. * (288. * N + 176. * N.powu(2) + 16. * N.powu(4))
        )
        / (3. * (1. + N).powu(4) * (2. + N))
        + 12. * Sm3e * (2. + N + N.powu(2)) / (N * (1. + N) * (2. + N))
        - 24. * Sm2e * (4. + N - N.powu(2)) / ((1. + N) * (2. + N)).powu(2)
        - S1
        * (48. / N + 432. + 564. * N + 324. * N.powu(2) + 138. * N.powu(3) + 48. * N.powu(4) + 6. * N.powu(5))
        / ((1. + N) * (2. + N)).powu(3)
        + S1
        * (-160. - 32. / N.powu(2) - 80. / N + 8. * N * (N - 1.))
        / (3. * (1. + N).powu(2) * (2. + N))
        - 6. * S1.powu(2) * (11. + 8. * N + N.powu(2) + 2. / N) / ((1. + N) * (2. + N)).powu(2)
        + 8. * S1.powu(2) * (2. / (3. * N) + 1.) / (N * (2. + N))
        - 2.
        * S2
        * (63. + 48. / N.powu(2) + 54. / N + 39. * N + 63. * N.powu(2) + 21. * N.powu(3))
        / ((N - 1.) * (1. + N).powu(2) * (2. + N).powu(2))
        + 8. * S2 * (17. - 2. / N.powu(2) - 5. / N + N * (17. + N)) / (3. * (1. + N).powu(2) * (2. + N))
        + (1. + 2. / N + N)
        / ((1. + N) * (2. + N))
        * (24. * Sm2e * S1 + 10. * S1.powu(3) / 9. + 46. * S1 * S2 / 3. + 176. * S3 / 9. - 24. * Sm21e);

    #[rustfmt::skip]
    let mut a_hg_l1 =
        2.
        * (
            640. + 2192. * N
            + 2072. * N.powu(2)
            + 868. * N.powu(3)
            + 518. * N.powu(4)
            + 736. * N.powu(5)
            + 806. * N.powu(6)
            + 542. * N.powu(7)
            + 228. * N.powu(8)
            + 38. * N.powu(9)
        )
        / (3. * (N * (N + 1.) * (N + 2.)).powu(3) * (N - 1.));

    a_hg_l1 -= 2.
        * (N * (N.powu(2) - 1.)
            * (N + 2.)
            * (4. * (36. + N * (88. + N * (33. + N * (8. + 9. * N)))) * S1m
                + N * (N + 1.)
                    * (N + 2.)
                    * (2. + N + N.powu(2))
                    * (10. * S1m.powu(2) + 18. * (2. * Sm2e + ZETA2) + 26. * S2m)))
        / (3. * (N * (N + 1.) * (N + 2.)).powu(3) * (N - 1.));

    a_hg_l1 += 12. * ZETA2 * (-2. + N + N.powu(3)) / (N * (N.powu(2) - 1.) * (N + 2.));

    #[rustfmt::skip]
    let a_hg_l2 = (
        4.
        * (2. + N + N.powu(2))
        * (2. * (-11. + N + N.powu(2)) * (1. + N + N.powu(2)) / (N - 1.))
        / (3. * (N * (N + 1.) * (N + 2.)).powu(2))
    ) + 20. * (2. + N + N.powu(2)) * S1 / (3. * N * (N + 1.) * (N + 2.));

    a_hg_l2 * L.pow(2) + a_hg_l1 * L + a_hg_l0
}
