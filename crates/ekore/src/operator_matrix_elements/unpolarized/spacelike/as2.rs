//! |NNLO| |QCD|.

use num::complex::Complex;
use num::traits::Pow;
use num::Zero;

use crate::constants::{CA, CF, TR, ZETA2, ZETA3};
use crate::harmonics::cache::{Cache, K};

use crate::operator_matrix_elements::unpolarized::spacelike::as1;

/// |NNLO| light-light non-singlet |OME|.
///
/// Implements Eq. (B.4) of [\[Buza:1996wv\]](crate::bib::Buza1996wv).
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

/// |NNLO| heavy-light pure-singlet |OME|.
///
/// Implements Eq. (B.1) of [\[Buza:1996wv\]](crate::bib::Buza1996wv).
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

/// |NNLO| heavy-gluon |OME|.
///
/// Implements Eq. (B.3) of [\[Buza:1996wv\]](crate::bib::Buza1996wv).
/// The expession for ``A_Hg_l0`` comes form [\[Bierenbaum:2009zt\]](crate::bib::Bierenbaum2009zt).
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

/// |NNLO| gluon-quark |OME|.
///
/// Implements Eq. (B.5) of [\[Buza:1996wv\]](crate::bib::Buza1996wv).
pub fn A_gq(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let S1m = S1 - 1. / N;

    let B2M = ((S1 - 1.0 / N).powu(2) + S2 - 1.0 / N.powu(2)) / (N - 1.0);
    let B21 = ((S1 + 1.0 / (N + 1.0)).powu(2) + S2 + 1.0 / (N + 1.0).powu(2)) / (N + 1.0);

    let a_gq_l0 = 4.0 / 3.0 * (2.0 * B2M - 2.0 * (S1.powu(2) + S2) / N + B21)
        + 8.0 / 9.0
            * (-10.0 * (S1 - 1.0 / N) / (N - 1.0) + 10.0 * S1 / N
                - 8.0 * (S1 + 1.0 / (N + 1.0)) / (N + 1.0))
        + 1.0 / 27.0 * (448.0 * (1.0 / (N - 1.0) - 1.0 / N) + 344.0 / (N + 1.0));

    let a_gq_l1 = -(-96.0 + 16.0 * N * (7.0 + N * (21.0 + 10.0 * N + 8.0 * N.powu(2)))
        - 48.0 * N * (1.0 + N) * (2.0 + N + N.powu(2)) * S1m)
        / (9.0 * (N - 1.0) * (N * (1.0 + N)).powu(2));

    let a_gq_l2 = 8. * (2. + N + N.powu(2)) / (3. * N * (N.powu(2) - 1.));

    CF * TR * (a_gq_l2 * L.pow(2) + a_gq_l1 * L + a_gq_l0)
}

/// |NNLO| gluon-gluon |OME|.
///
/// Implements Eq. (B.7) of [\[Buza:1996wv\]](crate::bib::Buza1996wv).
pub fn A_gg(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S1m = S1 - 1. / N;

    let D1 = -1.0 / N.powu(2);
    let D11 = -1.0 / (N + 1.0).powu(2);
    let D2 = 2.0 / N.powu(3);
    let D21 = 2.0 / (N + 1.0).powu(3);

    let a_gg_f = -15.0 - 8.0 / (N - 1.0) + 80.0 / N - 48.0 / (N + 1.0) - 24.0 / (N + 2.0)
        + 4.0 / 3.0 * (-6.0 / N.powu(4) - 6.0 / (N + 1.0).powu(4))
        + 6.0 * D2
        + 10.0 * D21
        + 32.0 * D1
        + 48.0 * D11;

    let a_gg_a = -224.0 / 27.0 * (S1 - 1.0 / N)
        + 10.0 / 9.0
        + 4.0 / 3.0 * (S1 + 1.0 / (N + 1.0)) / (N + 1.0)
        + 1.0 / 27.0 * (556.0 / (N - 1.0) - 628.0 / N + 548.0 / (N + 1.0) - 700.0 / (N + 2.0))
        + 4.0 / 3.0 * (D2 + D21)
        + 1.0 / 9.0 * (52.0 * D1 + 88.0 * D11);

    let a_gg_l0 = TR * (CF * a_gg_f + CA * a_gg_a);

    let a_gg_l1 = 8. / 3.
        * ((8. + 2. * N
            - 34. * N.powu(2)
            - 72. * N.powu(3)
            - 77. * N.powu(4)
            - 37. * N.powu(5)
            - 19. * N.powu(6)
            - 11. * N.powu(7)
            - 4. * N.powu(8))
            / ((N * (N + 1.)).powu(3) * (-2. + N + N.powu(2)))
            + 5. * S1m);

    let a_gg_l2 = 4. / 9.
        * (1.
            + 6. * (2. + N + N.powu(2)).powu(2) / ((N * (N + 1.)).powu(2) * (-2. + N + N.powu(2)))
            - 9. * (-4. - 3. * N + N.powu(3)) / (N * (N + 1.) * (-2. + N + N.powu(2))))
        - 4. * S1m;

    a_gg_l2 * L.pow(2) + a_gg_l1 * L + a_gg_l0
}

/// |NNLO| singlet |OME|.
pub fn A_singlet(c: &mut Cache, nf: u8, L: f64, is_msbar_mass: bool) -> [[Complex<f64>; 3]; 3] {
    let A_hq_2 = A_hq_ps(c, nf, L);
    let A_qq_2 = A_qq_ns(c, nf, L);
    let mut A_hg_2 = A_hg(c, nf, L);
    let A_gq_2 = A_gq(c, nf, L);
    let mut A_gg_2 = A_gg(c, nf, L);

    if is_msbar_mass {
        A_hg_2 -= 2.0 * 4.0 * CF * as1::A_hg(c, nf, 1.0);
        A_gg_2 -= 2.0 * 4.0 * CF * as1::A_gg(c, nf, 1.0);
    }

    [
        [A_gg_2, A_gq_2, Complex::<f64>::zero()],
        [Complex::<f64>::zero(), A_qq_2, Complex::<f64>::zero()],
        [A_hg_2, A_hq_2, Complex::<f64>::zero()],
    ]
}

/// |NNLO| non-singlet |OME|.
pub fn A_ns(c: &mut Cache, nf: u8, L: f64) -> [[Complex<f64>; 2]; 2] {
    [
        [A_qq_ns(c, nf, L), Complex::<f64>::zero()],
        [Complex::<f64>::zero(), Complex::<f64>::zero()],
    ]
}

#[cfg(test)]
mod test {
    use crate::cmplx;
    use crate::{
        harmonics::cache::Cache, operator_matrix_elements::unpolarized::spacelike::as2::*,
    };
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;
    const NF: u8 = 5;

    #[test]
    fn test_quark_number_conservation() {
        let logs = [0., 100.];
        for L in logs {
            let N = cmplx![1., 0.];
            let mut c = Cache::new(N);
            let aNSqq2 = A_qq_ns(&mut c, NF, L);
            assert_approx_eq!(f64, aNSqq2.re, 0.0, epsilon = 2e-11);
            assert_approx_eq!(f64, aNSqq2.im, 0.0, epsilon = 2e-11);
        }
    }

    #[test]
    fn test_momentum_conservation() {
        let logs = [0., 100.];
        for L in logs {
            let N = cmplx![2., 0.];
            let mut c = Cache::new(N);
            let aS2 = A_singlet(&mut c, NF, L, false);

            // gluon momenum conservation
            assert_approx_eq!(
                f64,
                (aS2[0][0] + aS2[1][0] + aS2[2][0]).re,
                0.0,
                epsilon = 2e-6
            );
            assert_approx_eq!(
                f64,
                (aS2[0][0] + aS2[1][0] + aS2[2][0]).im,
                0.0,
                epsilon = 2e-6
            );

            // quark momentum conservation
            // assert_approx_eq!(f64, (aS2[0][1] + aS2[1][1] + aS2[2][1]).re, 0.0, epsilon = 1e-11);
            // assert_approx_eq!(f64, (aS2[0][1] + aS2[1][1] + aS2[2][1]).im, 0.0, epsilon = 1e-11);
        }
    }
}
