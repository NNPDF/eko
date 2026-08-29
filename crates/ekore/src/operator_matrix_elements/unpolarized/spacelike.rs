//! The unpolarized, space-like |OME|.
use crate::constants::MAX_ORDER_QCD;
use crate::harmonics::cache::Cache;
use num::Zero;
use num::complex::Complex;
mod as1;
mod as2;
mod as3;

/// Compute the tower of the unpolarized, space-like singlet |OME|.
///
/// This function computes the first `matching_order_qcd` entries at the Mellin variable set in `cache`
/// using `nf` light flavors and `L`, logarithm of the squared ratio of factorization scale and mass
/// $\ln(\mu_F^2 / m^2)$.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 1, 3, 3)`. Only the first `matching_order_qcd`
/// entries along the outer axis are filled; remaining slots are zero.
///
/// # Available perturbative orders:
///
/// - $a_s^1$ [\[Ball:2015tna\]](crate::bib::Ball2015tna) [\[Buza:1996wv\]](crate::bib::Buza1996wv)
/// - $a_s^2$ [\[Buza:1996wv\]](crate::bib::Buza1996wv) [\[Bierenbaum:2009zt\]](crate::bib::Bierenbaum2009zt)
/// - $a_s^3$ (via external `libome`)
///
/// # Panics
///
/// Panics if `matching_order_qcd` is larger than the currently available order.
pub fn A_singlet(
    matching_order_qcd: usize,
    cache: &mut Cache,
    nf: u8,
    L: f64,
) -> [[[Complex<f64>; 3]; 3]; MAX_ORDER_QCD - 1] {
    if matching_order_qcd >= 4 {
        panic!("OME beyond N3LO is not yet implemented");
    }
    let mut A_s = [[[Complex::<f64>::zero(); 3]; 3]; MAX_ORDER_QCD - 1];
    if matching_order_qcd >= 1 {
        A_s[0] = as1::A_singlet(cache, nf, L);
    }
    if matching_order_qcd >= 2 {
        // TODO recover MSbar mass
        A_s[1] = as2::A_singlet(cache, nf, L, false);
    }
    if matching_order_qcd >= 3 {
        A_s[2] = as3::A_singlet(cache, nf, L);
    }
    A_s
}

/// Compute the tower of the unpolarized, space-like non-singlet |OME|.
///
/// This function computes the first `matching_order_qcd` entries at the Mellin variable set in `cache`
/// using `nf` light flavors and `L`, logarithm of the squared ratio of factorization scale and mass
/// $\ln(\mu_F^2 / m^2)$.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 1, 2, 2)`. Only the first `matching_order_qcd`
/// entries along the outer axis are filled; remaining slots are zero.
///
/// # Available perturbative orders:
///
/// - $a_s^1$ [\[Ball:2015tna\]](crate::bib::Ball2015tna)
/// - $a_s^2$ [\[Buza:1996wv\]](crate::bib::Buza1996wv)
/// - $a_s^3$ (via external `libome`)
///
/// # Panics
///
/// Panics if `matching_order_qcd` is larger than the currently available order.
pub fn A_non_singlet(
    matching_order_qcd: usize,
    cache: &mut Cache,
    nf: u8,
    L: f64,
) -> [[[Complex<f64>; 2]; 2]; MAX_ORDER_QCD - 1] {
    if matching_order_qcd >= 4 {
        panic!("OME beyond N3LO is not yet implemented");
    }
    let mut A_ns = [[[Complex::<f64>::zero(); 2]; 2]; MAX_ORDER_QCD - 1];
    if matching_order_qcd >= 1 {
        A_ns[0] = as1::A_ns(cache, nf, L);
    }
    if matching_order_qcd >= 2 {
        A_ns[1] = as2::A_ns(cache, nf, L);
    }
    if matching_order_qcd >= 3 {
        A_ns[2] = as3::A_ns(cache, nf, L);
    }
    A_ns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, assert_approx_eq_cmplx_2d, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_shapes() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(0., 1.);
        const L: f64 = 0.0;
        for matching_order_qcd in 1..=3usize {
            let mut cache = Cache::new(N);
            let a_s = A_singlet(matching_order_qcd, &mut cache, NF, L);
            assert_eq!(a_s.len(), MAX_ORDER_QCD - 1);
            assert_eq!(a_s[0].len(), 3);
            assert_eq!(a_s[0][0].len(), 3);
            // slots beyond order_qcd must be zero
            for item in a_s.iter().skip(matching_order_qcd) {
                assert_approx_eq_cmplx_2d!(f64, item, [[cmplx!(0., 0.); 3]; 3], 3);
            }
            let a_ns = A_non_singlet(matching_order_qcd, &mut cache, NF, L);
            assert_eq!(a_ns.len(), MAX_ORDER_QCD - 1);
            assert_eq!(a_ns[0].len(), 2);
            assert_eq!(a_ns[0][0].len(), 2);
            // slots beyond order_qcd must be zero
            for item in a_ns.iter().skip(matching_order_qcd) {
                assert_approx_eq_cmplx_2d!(f64, item, [[cmplx!(0., 0.); 2]; 2], 2);
            }
        }
    }

    #[test]
    fn test_a_non_singlet() {
        // reference combinations from the individual as1/as2 modules
        const NF: u8 = 5;
        const N: Complex<f64> = cmplx!(1., 0.);
        const L: f64 = 0.0;
        let mut cache = Cache::new(N);
        let a_ns = A_non_singlet(3, &mut cache, NF, L);
        // LO
        assert_approx_eq_cmplx_2d!(f64, a_ns[0], [[cmplx!(0., 0.); 2]; 2], 2, epsilon = 1e-14);
        // NNLO
        assert_approx_eq_cmplx_2d!(f64, a_ns[1], [[cmplx!(0., 0.); 2]; 2], 2, epsilon = 1e-14);
        // N3LO
        assert_approx_eq_cmplx_2d!(f64, a_ns[2], [[cmplx!(0., 0.); 2]; 2], 2, epsilon = 1e-14);
    }

    #[test]
    fn test_a_singlet() {
        // reference combinations from the individual as1/as2 modules
        const NF: u8 = 5;
        const N: Complex<f64> = cmplx!(2., 0.);
        const L: f64 = 100.;
        let mut cache = Cache::new(N);
        let a_s = A_singlet(3, &mut cache, NF, L);
        // LO
        assert_approx_eq_cmplx!(
            f64,
            a_s[0][0][2] + a_s[0][1][2] + a_s[0][2][2],
            Complex::zero(),
            epsilon = 1e-10
        );
        assert_approx_eq_cmplx!(
            f64,
            a_s[0][0][0] + a_s[0][1][0] + a_s[0][2][0],
            Complex::zero()
        );
        // NNLO
        assert_approx_eq_cmplx!(
            f64,
            a_s[1][0][0] + a_s[1][1][0] + a_s[1][2][0],
            Complex::zero(),
            epsilon = 2e-6
        );
        assert_approx_eq_cmplx!(
            f64,
            a_s[1][0][1] + a_s[1][1][1] + a_s[1][2][1],
            Complex::zero(),
            epsilon = 1e-11
        );
        // N3LO
        assert_approx_eq_cmplx_2d!(f64, a_s[2], [[cmplx!(0., 0.); 3]; 3], 3, epsilon = 1e-14);
    }

    #[test]
    #[should_panic(expected = "OME beyond N3LO is not yet implemented")]
    fn test_a_singlet_order4_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        const L: f64 = 0.0;
        let mut cache = Cache::new(N);
        A_singlet(4, &mut cache, NF, L);
    }

    #[test]
    #[should_panic(expected = "OME beyond N3LO is not yet implemented")]
    fn test_a_non_singlet_order4_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        const L: f64 = 0.0;
        let mut cache = Cache::new(N);
        A_non_singlet(4, &mut cache, NF, L);
    }
}
