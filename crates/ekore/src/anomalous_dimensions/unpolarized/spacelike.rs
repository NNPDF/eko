//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::constants::{
    ed2, eu2, PID_NSM, PID_NSM_ED2, PID_NSM_EU2, PID_NSP, PID_NSP_ED2, PID_NSP_EU2, PID_NSV,
};
use crate::harmonics::cache::Cache;
use num::complex::Complex;
use num::Zero;
pub mod aem1;
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
    let row = vec![Complex::<f64>::zero(); order_qcd + 1];
    let mut gamma_ns = vec![row; order_qed + 1];
    gamma_ns[1][0] = as1::gamma_ns(c, nf);
    gamma_ns[0][1] = choose_ns_as_aem1(mode, c, nf);
    gamma_ns
}

pub fn choose_ns_as_aem1(mode: u16, c: &mut Cache, nf: u8) -> Complex<f64> {
    match mode {
        PID_NSP_EU2 | PID_NSM_EU2 => eu2 * aem1::gamma_ns(c, nf),
        PID_NSP_ED2 | PID_NSM_ED2 => ed2 * aem1::gamma_ns(c, nf),
        _ => panic!("Unkown non-singlet sector element"),
    }
}
