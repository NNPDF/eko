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

/// Compute the valence-like non-singlet anomalous dimension.
///
/// The routine is taken from [\[Moch:2017uml\]][crate::bib::Moch2017uml].
///
/// The $n_f^{0,1}$ leading large-$N_c$ contributions and the $n_f^2$ part
/// are high-accuracy (0.1% or better) parametrizations of the exact
/// results. The $n_f^3$ expression is exact up to numerical truncations.
///
/// The remaining $n_f^{0,1}$ terms are approximations based on the first
/// eight even moments together with small-x and large-x constraints.
/// The two sets spanning the error estimate are called via `variation = 1`
/// and  `variation = 2`. Any other value of `variation` invokes their average.
pub use gnsv::gamma_nsv;

pub use gps::gamma_ps;
pub use gqg::gamma_qg;

// Compute the singlet anomalous dimension matrix.
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
