//! |LO| |QCD|.

use num::complex::Complex;

use super::super::unpolarized::spacelike::as1::gamma_ns;
use crate::constants::{CA, CF, TR};
use crate::harmonics::cache::{Cache, K};

/// Compute the quark-gluon anomalous dimension.
///
/// Implements Eq. (A.1) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_qg(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let gamma = -(N - 1) / N / (N + 1);
    2.0 * TR * 2.0 * nf * gamma
}

/// Compute the gluon-quark anomalous dimension.
///
/// Implements Eq. (A.1) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_gq(c: &mut Cache, _nf: u8) -> Complex<f64> {
    let N = c.n();
    let gamma = -(N + 2) / N / (N + 1);
    2.0 * CF * gamma
}

/// Compute the gluon-gluon anomalous dimension.
///
/// Implements Eq. (A.1) of [\[Gluck:1995yr\]][crate::bib::Gluck1995yr].
pub fn gamma_gg(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let gamma = -S1 + 2 / N / (N + 1);
    CA * (-4.0 * gamma - 11.0 / 3.0) + 4.0 / 3.0 * TR * nf;
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let gamma_qq = gamma_ns(c, nf);
    [
        [gamma_qq, gamma_qg(c, nf)],
        [gamma_gq(c, nf), gamma_gg(c, nf)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmonics::cache::Cache;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;
    const NF: u8 = 5;

    #[test]
    fn quark_momentum_conservation() {
        const N: Complex<f64> = cmplx![2., 0.];
        let mut c = Cache::new(N);
        let me = gamma_ns(&mut c, NF) + gamma_gq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn gluon_momentum_conservation() {
        const N: Complex<f64> = cmplx![2., 0.];
        let mut c = Cache::new(N);
        let me = gamma_qg(&mut c, NF) + gamma_gg(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn qg_helicity_conservation() {
        const N: Complex<f64> = cmplx![1., 0.];
        let mut c = Cache::new(N);
        let me = gamma_qg(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::Zero(), epsilon = 1e-12);
    }
}
