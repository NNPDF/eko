//! The unpolarized, space-like |OME| at various couplings power.
use crate::harmonics::cache::Cache;
use num::complex::Complex;
use num::Zero;
pub mod as1;
pub mod as2;

/// Compute the tower of the singlet |OME|.
pub fn A_singlet(
    matching_order_qcd: usize,
    c: &mut Cache,
    nf: u8,
    L: f64,
) -> Vec<[[Complex<f64>; 3]; 3]> {
    let mut A_s = vec![
        [
            [
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero()
            ],
            [
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero()
            ],
            [
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero()
            ]
        ];
        matching_order_qcd
    ];
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
pub fn A_non_singlet(
    matching_order_qcd: usize,
    c: &mut Cache,
    nf: u8,
    L: f64,
) -> Vec<[[Complex<f64>; 2]; 2]> {
    let mut A_ns = vec![
        [
            [Complex::<f64>::zero(), Complex::<f64>::zero()],
            [Complex::<f64>::zero(), Complex::<f64>::zero()]
        ];
        matching_order_qcd
    ];
    if matching_order_qcd >= 1 {
        A_ns[0] = as1::A_ns(c, nf, L);
    }
    if matching_order_qcd >= 2 {
        A_ns[1] = as2::A_ns(c, nf, L);
    }
    A_ns
}
