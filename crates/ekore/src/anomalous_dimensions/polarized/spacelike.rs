//! The polarized, space-like anomalous dimensions.

use crate::constants::{MAX_ORDER_QCD, PID_NSM, PID_NSP, PID_NSV};
use crate::harmonics::cache::Cache;
use num::Zero;
use num::complex::Complex;
mod as1;
mod as2;
// pub mod as3;

/// Compute the tower of the polarized, space-like non-singlet anomalous dimensions.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 2,)`. Only the first `order_qcd` entries
/// are filled; remaining slots are zero.
///
/// # Available perturbative orders:
///
/// - $a_s^1$ [\[Moch:2004pa\]][crate::bib::Moch2004pa]
/// - $a_s^2$ [\[Moch:2004pa\]][crate::bib::Moch2004pa]
///
/// # Panics
///
/// Panics if `order_qcd` is larger than the currently available order or `mode` is not a valid |PID|.
pub fn gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    cache: &mut Cache,
    nf: u8,
    _n3lo_variation: [u8; 3],
) -> [Complex<f64>; MAX_ORDER_QCD - 2] {
    if order_qcd >= 3 {
        panic!("Polarized beyond NLO is not yet implemented");
    }
    let mut gamma_ns = [Complex::<f64>::zero(); MAX_ORDER_QCD - 2];
    gamma_ns[0] = as1::gamma_ns(cache, nf);
    // NLO and beyond
    if order_qcd >= 2 {
        let gamma_ns_1 = match mode {
            PID_NSP => as2::gamma_nsp(cache, nf),
            // To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
            PID_NSM | PID_NSV => as2::gamma_nsm(cache, nf),
            _ => panic!("Unkown non-singlet sector element"),
        };
        gamma_ns[1] = gamma_ns_1
    }
    // // NNLO and beyond
    // if order_qcd >= 3 {
    //     let gamma_ns_2 = match mode {
    //         PID_NSP => as3::gamma_nsp(cache, nf),
    //         PID_NSM => as3::gamma_nsm(cache, nf),
    //         PID_NSV => as3::gamma_nsv(cache, nf),
    //         _ => panic!("Unkown non-singlet sector element"),
    //     };
    //     gamma_ns[2] = gamma_ns_2
    // }
    gamma_ns
}

/// Compute the tower of the polarized, space-like singlet anomalous dimension matrices.
///
/// Returns an array of shape `(MAX_ORDER_QCD - 2, 2, 2)`. Only the first `order_qcd`
/// entries along the outer axis are filled; remaining slots are zero.
///
/// # Available perturbative orders:
///
/// - $a_s^1$ [\[Gluck:1995yr\]][crate::bib::Gluck1995yr]
/// - $a_s^2$ [\[Gluck:1995yr\]][crate::bib::Gluck1995yr] [\[Moch:2004pa\]][crate::bib::Moch2004pa]
///
/// # Panics
///
/// Panics if `order_qcd` is larger than the currently available order.
pub fn gamma_singlet_qcd(
    order_qcd: usize,
    cache: &mut Cache,
    nf: u8,
    _n3lo_variation: [u8; 4],
) -> [[[Complex<f64>; 2]; 2]; MAX_ORDER_QCD - 2] {
    if order_qcd >= 3 {
        panic!("Polarized beyond NLO is not yet implemented");
    }
    let mut gamma_S = [[[Complex::<f64>::zero(); 2]; 2]; MAX_ORDER_QCD - 2];
    gamma_S[0] = as1::gamma_singlet(cache, nf);
    // NLO and beyond
    if order_qcd >= 2 {
        gamma_S[1] = as2::gamma_singlet(cache, nf);
    }
    // // NNLO and beyond
    // if order_qcd >= 3 {
    //     gamma_S[2] = as3::gamma_singlet(cache, nf);
    // }
    gamma_S
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_approx_eq_cmplx, assert_approx_eq_cmplx_1d, assert_approx_eq_cmplx_2d, cmplx,
    };
    use num::complex::Complex;

    #[test]
    fn test_shapes() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(2., 0.);
        for order_qcd in 1..=2usize {
            let mut cache = Cache::new(N);
            let gamma_ns = gamma_ns_qcd(order_qcd, PID_NSP, &mut cache, NF, [0u8; 3]);
            assert_eq!(gamma_ns.len(), MAX_ORDER_QCD - 2);
            // slots beyond order_qcd must be zero
            for item in gamma_ns.iter().skip(order_qcd) {
                assert_approx_eq_cmplx!(f64, *item, cmplx!(0., 0.));
            }
            let gamma_s = gamma_singlet_qcd(order_qcd, &mut cache, NF, [0u8; 4]);
            assert_eq!(gamma_s.len(), MAX_ORDER_QCD - 2);
            assert_eq!(gamma_s[0].len(), 2);
            assert_eq!(gamma_s[0][0].len(), 2);
            // slots beyond order_qcd must be zero
            for item in gamma_s.iter().skip(order_qcd) {
                assert_approx_eq_cmplx_2d!(f64, item, [[cmplx!(0., 0.); 2]; 2], 2);
            }
        }
    }

    #[test]
    fn test_gamma_ns_qcd() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut cache = Cache::new(N);
        let n3lo_variation = [0u8; 3];

        // LO
        assert_approx_eq_cmplx!(
            f64,
            gamma_ns_qcd(2, PID_NSP, &mut cache, NF, n3lo_variation)[0],
            cmplx!(0., 0.),
            epsilon = 1e-14
        );

        // NLO
        assert_approx_eq_cmplx_1d!(
            f64,
            gamma_ns_qcd(2, PID_NSP, &mut cache, NF, n3lo_variation),
            [cmplx!(0., 0.); 2],
            2,
            epsilon = 2e-6
        );
    }

    #[test]
    fn test_gamma_singlet_qcd() {
        // reference combinations from the individual as1/as2 modules
        use crate::constants::{CA, CF, TR};
        use num::traits::Pow;
        use std::f64::consts::PI;

        const NF: u8 = 5;
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut cache = Cache::new(N);
        let gamma_s = gamma_singlet_qcd(2, &mut cache, NF, [0u8; 4]);

        // LO
        assert_approx_eq_cmplx!(
            f64,
            gamma_s[0][0][0] + gamma_s[0][1][0],
            cmplx!(4. * CF / 3., 0.),
            epsilon = 1e-12
        );
        assert_approx_eq_cmplx!(
            f64,
            gamma_s[0][0][1] + gamma_s[0][1][1],
            cmplx!(3. + (NF as f64) / 3., 0.),
            epsilon = 1e-12
        );

        // NLO
        assert_approx_eq_cmplx!(
            f64,
            -gamma_s[1][0][1],
            cmplx!(
                4. * (NF as f64)
                    * (0.574074074 * CF - 2. * CA * (-7. / 18. + 1. / 6. * (5. - PI.pow(2) / 3.)))
                    * TR,
                0.
            ),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "Unkown non-singlet sector element")]
    fn test_unknown_pid_panics() {
        use crate::constants::PID_NSM_U;
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        let mut cache = Cache::new(N);
        // PID_NSM_U (10202) is not a valid mode for polarized non-singlet
        gamma_ns_qcd(2, PID_NSM_U, &mut cache, NF, [0u8; 3]);
    }

    #[test]
    #[should_panic(expected = "Polarized beyond NLO is not yet implemented")]
    fn test_gamma_ns_order4_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(1.234, 0.);
        let mut cache = Cache::new(N);
        gamma_ns_qcd(3, PID_NSM, &mut cache, NF, [0u8; 3]);
    }

    #[test]
    #[should_panic(expected = "Polarized beyond NLO is not yet implemented")]
    fn test_gamma_singlet_order4_panics() {
        const NF: u8 = 4;
        const N: Complex<f64> = cmplx!(2.345, 0.);
        let mut cache = Cache::new(N);
        gamma_singlet_qcd(3, &mut cache, NF, [0u8; 4]);
    }
}
