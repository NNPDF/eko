use crate::harmonics::cache::Cache;
use num::complex::Complex;
use num::Zero;
mod as1;

/// Compute the tower of the non-singlet anomalous dimensions.
pub fn gamma_ns(order_qcd: usize, _mode: u16, c: &mut Cache, nf: u8) -> Vec<Complex<f64>> {
    let mut gamma_ns = vec![Complex::<f64>::zero(); order_qcd];
    gamma_ns[0] = as1::gamma_ns(c, nf);
    gamma_ns
}

/// Compute the tower of the singlet anomalous dimension matrices.
pub fn gamma_singlet(order_qcd: usize, c: &mut Cache, nf: u8) -> Vec<[[Complex<f64>; 2]; 2]> {
    let mut gamma_S = vec![
        [
            [Complex::<f64>::zero(), Complex::<f64>::zero()],
            [Complex::<f64>::zero(), Complex::<f64>::zero()]
        ];
        order_qcd
    ];
    gamma_S[0] = as1::gamma_singlet(c, nf);
    gamma_S
}
