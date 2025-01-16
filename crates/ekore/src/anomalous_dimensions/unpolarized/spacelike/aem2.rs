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

/// Compute the non-singlet anomalous dimension.
///
/// Implements singlet part of Eq. (57) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
fn gamma_nsq(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    let S2 = c.get(K::S2);
    let gamma = 2.0 * (-12.0 + 20.0 * N + 47.0 * N.powu(2) + 6.0 * N.powu(3) + 3.0 * N.powu(4))
        / (9.0 * N.powu(2) * (1.0 + N).powu(2))
        - 80.0 / 9.0 * S1
        + 16.0 / 3.0 * S2;
    let cc = ChargeCombinations { nf };
    let eSigma2 = (NC as f64) * cc.e2avg() * (nf as f64);
    eSigma2 * gamma
}

/// Compute the singlet-like non-singlet up anomalous dimension.
///
/// Implements sum of Eqs. (57-58) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=u.
pub fn gamma_nspu(c: &mut Cache, nf: u8) -> Complex<f64> {
    EU2 * as1aem1::gamma_nsp(c, nf) / CF / 2. + gamma_nsq(c, nf)
}

/// Compute the singlet-like non-singlet down anomalous dimension.
///
/// Implements sum of Eqs. (57-58) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=d.
pub fn gamma_nspd(c: &mut Cache, nf: u8) -> Complex<f64> {
    ED2 * as1aem1::gamma_nsp(c, nf) / CF / 2. + gamma_nsq(c, nf)
}

/// Compute the valence-like non-singlet up anomalous dimension.
///
/// Implements difference between Eqs. (57-58) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=u.
pub fn gamma_nsmu(c: &mut Cache, nf: u8) -> Complex<f64> {
    EU2 * as1aem1::gamma_nsm(c, nf) / CF / 2. + gamma_nsq(c, nf)
}

/// Compute the valence-like non-singlet down anomalous dimension.
///
/// Implements difference between Eqs. (57-58) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk] for q=d.
pub fn gamma_nsmd(c: &mut Cache, nf: u8) -> Complex<f64> {
    ED2 * as1aem1::gamma_nsm(c, nf) / CF / 2. + gamma_nsq(c, nf)
}

/// Compute the pure-singlet anomalous dimension.
///
/// Implements Eq. (59) of [\[deFlorian:2016gvk\]][crate::bib::deFlorian2016gvk].
pub fn gamma_ps(c: &mut Cache, nf: u8) -> Complex<f64> {
    let N = c.n();
    let result = -4.0 * (2.0 + N * (5.0 + N)) * (4.0 + N * (4.0 + N * (7.0 + 5.0 * N)))
        / ((-1.0 + N) * N.powu(3) * (1.0 + N).powu(3) * (2.0 + N).powu(2));
    2. * (nf as f64) * CA * result
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 4]; 4] {
    let cc = ChargeCombinations { nf };
    const E2M: f64 = EU2 - ED2;

    let gamma_phu = gamma_phu(c, nf);
    let gamma_phd = gamma_phd(c, nf);
    let gamma_uph = gamma_uph(c, nf);
    let gamma_dph = gamma_dph(c, nf);
    let gamma_nspu = gamma_nspu(c, nf);
    let gamma_nspd = gamma_nspd(c, nf);
    let gamma_ps = gamma_ps(c, nf);

    [
        [
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
        ],
        [
            Complex::<f64>::zero(),
            gamma_phph(c, nf),
            cc.vu() * EU2 * gamma_phu + cc.vd() * ED2 * gamma_phd,
            cc.vu() * (EU2 * gamma_phu - ED2 * gamma_phd),
        ],
        [
            Complex::<f64>::zero(),
            cc.vu() * EU2 * gamma_uph + cc.vd() * ED2 * gamma_dph,
            cc.vu() * EU2 * gamma_nspu + cc.vd() * ED2 * gamma_nspd + cc.e2avg().powi(2) * gamma_ps,
            cc.vu() * (EU2 * gamma_nspu - ED2 * gamma_nspd + E2M * cc.e2avg() * gamma_ps),
        ],
        [
            Complex::<f64>::zero(),
            cc.vd() * (EU2 * gamma_uph - ED2 * gamma_dph),
            cc.vd() * (EU2 * gamma_nspu - ED2 * gamma_nspd + E2M * cc.e2avg() * gamma_ps),
            cc.vd() * EU2 * gamma_nspu
                + cc.vu() * ED2 * gamma_nspd
                + cc.vu() * cc.vd() * E2M.powi(2) * gamma_ps,
        ],
    ]
}

/// Compute the valence anomalous dimension matrix.
pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let cc = ChargeCombinations { nf };
    let gamma_nsmu = gamma_nsmu(c, nf);
    let gamma_nsmd = gamma_nsmd(c, nf);
    [
        [
            cc.vu() * EU2 * gamma_nsmu + cc.vd() * ED2 * gamma_nsmd,
            cc.vu() * (EU2 * gamma_nsmu - ED2 * gamma_nsmd),
        ],
        [
            cc.vd() * (EU2 * gamma_nsmu - ED2 * gamma_nsmd),
            cc.vd() * EU2 * gamma_nsmu + cc.vu() * ED2 * gamma_nsmd,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;
    use num::Zero;

    #[test]
    fn number_conservation() {
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        for nf in 2..7 {
            assert_approx_eq_cmplx!(f64, gamma_nsmu(&mut c, nf), Complex::zero(), epsilon = 1e-5);
            assert_approx_eq_cmplx!(f64, gamma_nsmd(&mut c, nf), Complex::zero(), epsilon = 1e-5);
        }
    }

    #[test]
    fn quark_momentum_conservation() {
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        for nf in 2..7 {
            let cc = ChargeCombinations { nf };
            let ps = EU2 * gamma_ps(&mut c, cc.nu()) + ED2 * gamma_ps(&mut c, cc.nd());
            let me_u = gamma_nspu(&mut c, nf) + ps + gamma_phu(&mut c, nf);
            assert_approx_eq_cmplx!(f64, me_u, Complex::zero(), epsilon = 1e-5);
            let me_d = gamma_nspd(&mut c, nf) + ps + gamma_phd(&mut c, nf);
            assert_approx_eq_cmplx!(f64, me_d, Complex::zero(), epsilon = 1e-5);
        }
    }

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
