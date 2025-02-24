//! |N3LO| |QCD|.

mod ggg;
mod ggq;
mod gnsm;
mod gnsp;
mod gnsv;
mod gps;
mod gqg;

use crate::harmonics::cache::Cache;
use num::complex::Complex;

pub use ggg::gamma_gg;
pub use ggq::gamma_gq;
pub use gnsm::gamma_nsm;
pub use gnsp::gamma_nsp;
pub use gnsv::gamma_nsv;
pub use gps::gamma_ps;
pub use gqg::gamma_qg;

/// Compute the singlet anomalous dimension matrix.
///
/// `variation = (gg, gq, qg, qq)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_singlet(c: &mut Cache, nf: u8, variation: [u8; 4]) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gnsp::gamma_nsp(c, nf, variation[3]) + gps::gamma_ps(c, nf, variation[3]);
    [
        [gamma_qq, gqg::gamma_qg(c, nf, variation[2])],
        [
            ggq::gamma_gq(c, nf, variation[1]),
            ggg::gamma_gg(c, nf, variation[0]),
        ],
    ]
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_momentum_conservation() {
        let NF = 5;
        let mut c = Cache::new(cmplx!(2., 0.));
        // Numbers are coming from the python implementation.
        let quark_refs: [f64; 3] = [0.053441, 0.225674, -0.118792];
        let gluon_refs: [f64; 3] = [-0.0300842, 0.283004, -0.343172];
        for imod in [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]] {
            let g_singlet = gamma_singlet(&mut c, NF, imod);
            // quark conservation
            assert_approx_eq_cmplx!(
                f64,
                g_singlet[0][0] + g_singlet[1][0],
                cmplx!(quark_refs[imod[0] as usize], 0.),
                rel = 2e-5
            );
            // gluon conservation
            assert_approx_eq_cmplx!(
                f64,
                g_singlet[0][1] + g_singlet[1][1],
                cmplx!(gluon_refs[imod[0] as usize], 0.),
                rel = 6e-5
            );
        }
    }
}
