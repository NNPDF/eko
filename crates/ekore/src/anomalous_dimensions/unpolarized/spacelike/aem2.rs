//! |NLO| |QED|.
use num::complex::Complex;
use num::Zero;

use crate::constants::{ChargeCombinations, CA, CF, ED2, EU2, NC};
use crate::harmonics::cache::{Cache, K};

use crate::anomalous_dimensions::unpolarized::spacelike::as1aem1;

/// Compute the photon-photon anomalous dimension.
///
/// Implements Eq. (68) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
pub fn gamma_phph(c: &mut Cache, nf: u8) -> Complex<f64> {
    let cc = ChargeCombinations { nf };
    (NC as f64)
        * ((cc.nu() as f64) * EU2.powi(2) + (cc.nd() as f64) * ED2.powi(2))
        * (as1aem1::gamma_gph(c, nf) / CF / CA + 4.)
}

/// Compute the up-photon anomalous dimension.
///
/// Implements Eq. (55) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=u.
pub fn gamma_uph(c: &mut Cache, nf: u8) -> Complex<f64> {
    EU2 * as1aem1::gamma_qph(c, nf) / CF
}

/// Compute the down-photon anomalous dimension.
///
/// Implements Eq. (55) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=d.
pub fn gamma_dph(c: &mut Cache, nf: u8) -> Complex<f64> {
    ED2 * as1aem1::gamma_qph(c, nf) / CF
}

/// Compute the photon-quark anomalous dimension.
///
/// Implements singlet part of Eq. (56) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
fn gamma_phq(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let gamma = (-16.0 * (-16.0 - 27.0 * N - 13.0 * N.powu(2) - 8.0 * N.powu(3)))
        / (9.0 * (-1.0 + N) * N * (1.0 + N).powu(2))
        - 16.0 * (2.0 + 3.0 * N + 2.0 * N.powu(2) + N.powu(3))
            / (3.0 * (-1.0 + N) * N * (1.0 + N).powu(2))
            * S1;
    let cc = ChargeCombinations { nf };
    let eSigma2 = (NC as f64) * cc.e2avg() * (nf as f64);
    eSigma2 * gamma
}

/// Compute the photon-up anomalous dimension.
///
/// Implements Eq. (56) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
pub fn gamma_phu(c: &mut Cache, nf: u8) -> Complex<f64> {
    EU2 * as1aem1::gamma_phq(c, nf) / CF + gamma_phq(c, nf)
}

/// Compute the photon-down anomalous dimension.
///
/// Implements Eq. (56) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
pub fn gamma_phd(c: &mut Cache, nf: u8) -> Complex<f64> {
    ED2 * as1aem1::gamma_phq(c, nf) / CF + gamma_phq(c, nf)
}

// /// Compute the non-singlet anomalous dimension.
// ///
// /// Implements Eq. (2.5) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
// pub fn gamma_ns(c: &mut Cache, nf: u8) -> Complex<f64> {
//     as1::gamma_ns(c, nf) / CF
// }

// /// Compute the singlet anomalous dimension matrix.
// pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 4]; 4] {
//     let cc = ChargeCombinations { nf };

//     let gamma_ph_q = gamma_phq(c, nf);
//     let gamma_q_ph = gamma_qph(c, nf);
//     let gamma_nonsinglet = gamma_ns(c, nf);

//     [
//         [
//             Complex::<f64>::zero(),
//             Complex::<f64>::zero(),
//             Complex::<f64>::zero(),
//             Complex::<f64>::zero(),
//         ],
//         [
//             Complex::<f64>::zero(),
//             gamma_phph(c, nf),
//             cc.e2avg() * gamma_ph_q,
//             cc.vue2m() * gamma_ph_q,
//         ],
//         [
//             Complex::<f64>::zero(),
//             cc.e2avg() * gamma_q_ph,
//             cc.e2avg() * gamma_nonsinglet,
//             cc.vue2m() * gamma_nonsinglet,
//         ],
//         [
//             Complex::<f64>::zero(),
//             cc.vde2m() * gamma_q_ph,
//             cc.vde2m() * gamma_nonsinglet,
//             cc.e2delta() * gamma_nonsinglet,
//         ],
//     ]
// }

// /// Compute the valence anomalous dimension matrix.
// pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
//     let cc = ChargeCombinations { nf };

//     [
//         [cc.e2avg() * gamma_ns(c, nf), cc.vue2m() * gamma_ns(c, nf)],
//         [cc.vde2m() * gamma_ns(c, nf), cc.e2delta() * gamma_ns(c, nf)],
//     ]
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;
    const NF: u8 = 5;

    // #[test]
    // fn number_conservation() {
    //     const N: Complex<f64> = cmplx!(1., 0.);
    //     let mut c = Cache::new(N);
    //     let me = gamma_ns(&mut c, NF);
    //     assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    // }

    // #[test]
    // fn quark_momentum_conservation() {
    //     const N: Complex<f64> = cmplx!(2., 0.);
    //     let mut c = Cache::new(N);
    //     let me = gamma_ns(&mut c, NF) + gamma_phq(&mut c, NF);
    //     assert_approx_eq_cmplx!(f64, me, Complex::zero(), epsilon = 1e-12);
    // }

    #[test]
    fn photon_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);

        for nf in 2..7 {
            let cc = ChargeCombinations { nf };
            assert_approx_eq_cmplx!(
                f64,
                EU2 * gamma_uph(&mut c, cc.nu())
                    + ED2 * gamma_dph(&mut c, cc.nd())
                    + gamma_phph(&mut c, nf),
                Complex::zero(),
                epsilon = 1e-12
            );
        }
    }
}
