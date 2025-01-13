//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::constants::{
    ED2, EU2, PID_NSM, PID_NSM_ED2, PID_NSM_EU2, PID_NSP, PID_NSP_ED2, PID_NSP_EU2, PID_NSV,
};
use crate::harmonics::cache::Cache;
use num::complex::Complex;
use num::Zero;
pub mod aem1;
pub mod aem2;
pub mod as1;
pub mod as1aem1;
pub mod as2;
pub mod as3;

/// Compute the tower of the non-singlet anomalous dimensions.
pub fn gamma_ns_qcd(order_qcd: usize, mode: u16, c: &mut Cache, nf: u8) -> Vec<Complex<f64>> {
    let mut gamma_ns = vec![Complex::<f64>::zero(); order_qcd];
    gamma_ns[0] = as1::gamma_ns(c, nf);
    // NLO and beyond
    if order_qcd >= 2 {
        let gamma_ns_1 = match mode {
            PID_NSP => as2::gamma_nsp(c, nf),
            // To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
            PID_NSM | PID_NSV => as2::gamma_nsm(c, nf),
            _ => panic!("Unkown non-singlet sector element"),
        };
        gamma_ns[1] = gamma_ns_1
    }
    // NNLO and beyond
    if order_qcd >= 3 {
        let gamma_ns_2 = match mode {
            PID_NSP => as3::gamma_nsp(c, nf),
            PID_NSM => as3::gamma_nsm(c, nf),
            PID_NSV => as3::gamma_nsv(c, nf),
            _ => panic!("Unkown non-singlet sector element"),
        };
        gamma_ns[2] = gamma_ns_2
    }
    gamma_ns
}

/// Compute the tower of the singlet anomalous dimension matrices.
pub fn gamma_singlet_qcd(order_qcd: usize, c: &mut Cache, nf: u8) -> Vec<[[Complex<f64>; 2]; 2]> {
    let mut gamma_S = vec![
        [
            [Complex::<f64>::zero(), Complex::<f64>::zero()],
            [Complex::<f64>::zero(), Complex::<f64>::zero()]
        ];
        order_qcd
    ];
    gamma_S[0] = as1::gamma_singlet(c, nf);
    // NLO and beyond
    if order_qcd >= 2 {
        gamma_S[1] = as2::gamma_singlet(c, nf);
    }
    // NNLO and beyond
    if order_qcd >= 3 {
        gamma_S[2] = as3::gamma_singlet(c, nf);
    }
    gamma_S
}

/// Compute the tower of the QED non-singlet anomalous dimensions.
pub fn gamma_ns_qed(
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    c: &mut Cache,
    nf: u8,
) -> Vec<Vec<Complex<f64>>> {
    let col = vec![Complex::<f64>::zero(); order_qcd + 1];
    let mut gamma_ns = vec![col; order_qed + 1];
    gamma_ns[1][0] = as1::gamma_ns(c, nf);
    gamma_ns[0][1] = choose_ns_as_aem1(mode, c, nf);
    gamma_ns[1][1] = choose_ns_as_as1aem1(mode, c, nf);
    gamma_ns
}

pub fn choose_ns_as_aem1(mode: u16, c: &mut Cache, nf: u8) -> Complex<f64> {
    match mode {
        PID_NSP_EU2 | PID_NSM_EU2 => EU2 * aem1::gamma_ns(c, nf),
        PID_NSP_ED2 | PID_NSM_ED2 => ED2 * aem1::gamma_ns(c, nf),
        _ => panic!("Unkown non-singlet sector element"),
    }
}

pub fn choose_ns_as_as1aem1(mode: u16, c: &mut Cache, nf: u8) -> Complex<f64> {
    match mode {
        PID_NSP_EU2 => EU2 * as1aem1::gamma_nsp(c, nf),
        PID_NSP_ED2 => ED2 * as1aem1::gamma_nsp(c, nf),
        PID_NSM_EU2 => EU2 * as1aem1::gamma_nsm(c, nf),
        PID_NSM_ED2 => ED2 * as1aem1::gamma_nsm(c, nf),
        _ => panic!("Unkown non-singlet sector element"),
    }
}

pub fn gamma_singlet_qed(
    order_qcd: usize,
    order_qed: usize,
    c: &mut Cache,
    nf: u8,
) -> Vec<Vec<[[Complex<f64>; 4]; 4]>> {
    let col = vec![
        [[
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero()
        ]; 4];
        order_qcd + 1
    ];

    let mut gamma_s = vec![col; order_qed + 1];

    gamma_s[1][0] = as1::gamma_singlet_qed(c, nf);
    gamma_s[0][1] = aem1::gamma_singlet(c, nf);
    gamma_s[1][1] = as1aem1::gamma_singlet(c, nf);
    gamma_s
}

/// Compute the grid of the QED valence anomalous dimensions matrices
pub fn gamma_valence_qed(
    order_qcd: usize,
    order_qed: usize,
    c: &mut Cache,
    nf: u8,
) -> Vec<Vec<[[Complex<f64>; 2]; 2]>> {
    let col = vec![[[Complex::<f64>::zero(), Complex::<f64>::zero(),]; 2]; order_qcd + 1];

    let mut gamma_v = vec![col; order_qed + 1];
    gamma_v[1][0] = as1::gamma_valence_qed(c, nf);
    gamma_v[0][1] = aem1::gamma_valence(c, nf);
    gamma_v[1][1] = as1aem1::gamma_valence(c, nf);
    gamma_v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn gamma_ns() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        assert_approx_eq_cmplx!(
            f64,
            gamma_ns_qcd(3, PID_NSP, &mut c, NF)[0],
            cmplx!(0., 0.),
            epsilon = 1e-14
        );

        for i in [0, 1] {
            assert_approx_eq_cmplx!(
                f64,
                gamma_ns_qcd(2, PID_NSM, &mut c, NF)[i],
                cmplx!(0., 0.),
                epsilon = 2e-6
            );
        }

        for i in 0..3 {
            assert_approx_eq_cmplx!(
                f64,
                gamma_ns_qcd(3, PID_NSM, &mut c, NF)[i],
                cmplx!(0., 0.),
                epsilon = 2e-4
            );
        }

        for i in 0..3 {
            assert_approx_eq_cmplx!(
                f64,
                gamma_ns_qcd(3, PID_NSV, &mut c, NF)[i],
                cmplx!(0., 0.),
                epsilon = 8e-4
            );
        }
    }

    #[test]
    fn test_gamma_ns_qed() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);

        for i in [0, 1] {
            for j in [0, 1] {
                assert_approx_eq_cmplx!(
                    f64,
                    gamma_ns_qed(1, 1, PID_NSM_EU2, &mut c, NF)[i][j],
                    cmplx!(0., 0.),
                    epsilon = 1e-5
                );
            }
        }

        for i in [0, 1] {
            for j in [0, 1] {
                assert_approx_eq_cmplx!(
                    f64,
                    gamma_ns_qed(1, 1, PID_NSM_ED2, &mut c, NF)[i][j],
                    cmplx!(0., 0.),
                    epsilon = 1e-5
                );
            }
        }

        assert_approx_eq_cmplx!(
            f64,
            gamma_ns_qed(1, 1, PID_NSP_EU2, &mut c, NF)[0][1],
            cmplx!(0., 0.),
            epsilon = 1e-5
        );

        assert_approx_eq_cmplx!(
            f64,
            gamma_ns_qed(1, 1, PID_NSP_ED2, &mut c, NF)[0][1],
            cmplx!(0., 0.),
            epsilon = 1e-5
        );
    }
}
