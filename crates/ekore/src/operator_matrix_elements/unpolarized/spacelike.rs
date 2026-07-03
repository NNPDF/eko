//! The unpolarized, space-like |OME| at various couplings power.
use crate::constants::MAX_ORDER_QCD;
use crate::harmonics::cache::Cache;
use num::Zero;
use num::complex::Complex;
mod as1;
mod as2;

/// Compute the tower of the singlet |OME|.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 1, d, d)`. Only the first `matching_order_qcd`
/// entries along the outer axis are filled; remaining slots are zero.
pub fn A_singlet(
    matching_order_qcd: usize,
    c: &mut Cache,
    nf: u8,
    L: f64,
) -> [[[Complex<f64>; 3]; 3]; MAX_ORDER_QCD - 1] {
    if matching_order_qcd >= 3 {
        panic!("OME beyond NLO is not yet implemented");
    }
    let mut A_s = [[[Complex::<f64>::zero(); 3]; 3]; MAX_ORDER_QCD - 1];
    if matching_order_qcd >= 1 {
        A_s[0] = as1::A_singlet(c, nf, L);
    }
    if matching_order_qcd >= 2 {
        // TODO recover MSbar mass
        A_s[1] = as2::A_singlet(c, nf, L, false);
    }
    A_s
}

/// Compute the tower of the non-singlet |OME|.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 1, d, d)`. Only the first `matching_order_qcd`
/// entries along the outer axis are filled; remaining slots are zero.
pub fn A_non_singlet(
    matching_order_qcd: usize,
    c: &mut Cache,
    nf: u8,
    L: f64,
) -> [[[Complex<f64>; 2]; 2]; MAX_ORDER_QCD - 1] {
    if matching_order_qcd >= 3 {
        panic!("OME beyond NLO is not yet implemented");
    }
    let mut A_ns = [[[Complex::<f64>::zero(); 2]; 2]; MAX_ORDER_QCD - 1];
    if matching_order_qcd >= 1 {
        A_ns[0] = as1::A_ns(c, nf, L);
    }
    if matching_order_qcd >= 2 {
        A_ns[1] = as2::A_ns(c, nf, L);
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
        for matching_order_qcd in 1..=2usize {
            let mut c = Cache::new(N);
            let a_s = A_singlet(matching_order_qcd, &mut c, NF, L);
            assert_eq!(a_s.len(), MAX_ORDER_QCD - 1);
            assert_eq!(a_s[0].len(), 3);
            assert_eq!(a_s[0][0].len(), 3);
            // slots beyond order_qcd must be zero
            for item in a_s.iter().skip(matching_order_qcd) {
                assert_approx_eq_cmplx_2d!(f64, item, [[cmplx!(0., 0.); 3]; 3], 3);
            }
            let a_ns = A_non_singlet(matching_order_qcd, &mut c, NF, L);
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
        let mut c = Cache::new(N);
        let a_ns = A_non_singlet(2, &mut c, NF, L);
        // LO
        assert_approx_eq_cmplx_2d!(f64, a_ns[0], [[cmplx!(0., 0.); 2]; 2], 2, epsilon = 1e-14);
        // NNLO
        assert_approx_eq_cmplx_2d!(f64, a_ns[1], [[cmplx!(0., 0.); 2]; 2], 2, epsilon = 1e-14);
    }

    #[test]
    fn test_momentum_conservation() {
        // reference combinations from the individual as1/as2 modules
        const NF: u8 = 5;
        const N: Complex<f64> = cmplx!(2., 0.);
        const L: f64 = 100.;
        let mut c = Cache::new(N);
        let a_s = A_singlet(2, &mut c, NF, L);
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
    }

    #[test]
    #[should_panic(expected = "OME beyond NLO is not yet implemented")]
    fn test_a_singlet_order3_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        const L: f64 = 0.0;
        let mut c = Cache::new(N);
        A_singlet(3, &mut c, NF, L);
    }

    #[test]
    #[should_panic(expected = "OME beyond NLO is not yet implemented")]
    fn test_a_non_singlet_order3_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        const L: f64 = 0.0;
        let mut c = Cache::new(N);
        A_non_singlet(3, &mut c, NF, L);
    }
}
