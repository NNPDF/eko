//! |LO| |QED|.
use num::complex::Complex;
use num::Zero;

use crate::constants::{ed2, eu2, uplike_flavors, ChargeCombinations, CF, NC, TR};
use crate::harmonics::cache::Cache;

use crate::anomalous_dimensions::unpolarized::spacelike::as1;

/// Compute the leading-order photon-quark anomalous dimension.
///
/// Implements Eq. (2.5) of
pub fn gamma_phq(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_gq(c, nf) / CF
}

/// Compute the leading-order quark-photon anomalous dimension.
///
/// Implements Eq. (2.5) of
pub fn gamma_qph(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_qg(c, nf) / TR * (NC as f64)
}

/// Compute the leading-order photon-photon anomalous dimension.
///
/// Implements Eq. (2.5) of
pub fn gamma_phph(_c: &mut Cache, nf: u8) -> Complex<f64> {
    let nu = uplike_flavors(nf);
    let nd = nf - nu;
    (4.0 / 3.0 * (NC as f64) * ((nu as f64) * eu2 + (nd as f64) * ed2)).into()
}

/// Compute the leading-order non-singlet QED anomalous dimension
///
/// Implements Eq. (2.5) of
pub fn gamma_ns(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_ns(c, nf) / CF
}

/// Compute the leading-order singlet QED anomalous dimension matrix
///
/// Implements Eq. (2.5) of
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> Vec<[Complex<f64>; 4]> {
    let cc = ChargeCombinations { nf };

    let gamma_ph_q = gamma_phq(c, nf);
    let gamma_q_ph = gamma_qph(c, nf);
    let gamma_nonsinglet = gamma_ns(c, nf);

    vec![
        [
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
        ],
        [
            Complex::<f64>::zero(),
            gamma_phph(c, nf),
            cc.e2avg() * gamma_ph_q,
            cc.vue2m() * gamma_ph_q,
        ],
        [
            Complex::<f64>::zero(),
            cc.e2avg() * gamma_q_ph,
            cc.e2avg() * gamma_nonsinglet,
            cc.vue2m() * gamma_nonsinglet,
        ],
        [
            Complex::<f64>::zero(),
            cc.vde2m() * gamma_q_ph,
            cc.vde2m() * gamma_nonsinglet,
            cc.e2delta() * gamma_nonsinglet,
        ],
    ]
}

/// Compute the leading-order valence QED anomalous dimension matrix
///
/// Implements Eq. (2.5) of
pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let cc = ChargeCombinations { nf };

    [
        [cc.e2avg() * gamma_ns(c, nf), cc.vue2m() * gamma_ns(c, nf)],
        [cc.vde2m() * gamma_ns(c, nf), cc.e2delta() * gamma_ns(c, nf)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;
    const NF: u8 = 5;

    #[test]
    fn number_conservation() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let me = gamma_ns(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn quark_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let me = gamma_ns(&mut c, NF) + gamma_phq(&mut c, NF);
        assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    }

    #[test]
    fn photon_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);

        for nf in 2..7 {
            let nu = uplike_flavors(nf);
            let nd = nf - nu;
            assert_approx_eq_cmplx!(
                f64,
                eu2 * gamma_qph(&mut c, nu) + ed2 * gamma_qph(&mut c, nd) + gamma_phph(&mut c, nf),
                cmplx!(0., 0.),
                epsilon = 2e-6
            );
        }
    }
}
