//! |LO| |QED|.
use num::complex::Complex;
use num::Zero;

use crate::cmplx;
use crate::constants::{ChargeCombinations, CF, ED2, EU2, NC, TR};
use crate::harmonics::cache::Cache;

use crate::anomalous_dimensions::unpolarized::spacelike::as1;

/// Compute the photon-quark anomalous dimension.
///
/// Implements Eq. (2.5) of [\[Carrazza:2015dea\]][crate::bib::Carrazza2015dea].
pub fn gamma_phq(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_gq(c, nf) / CF
}

/// Compute the quark-photon anomalous dimension.
///
/// Implements Eq. (2.5) of [\[Carrazza:2015dea\]][crate::bib::Carrazza2015dea].
/// However, we are adding the $N_C$ and the $2n_f$ factors from $\theta$ inside the
/// definition of $\gamma_{q \gamma}^{(0)}(N)$.
pub fn gamma_qph(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_qg(c, nf) / TR * (NC as f64)
}

/// Compute the photon-photon anomalous dimension.
///
/// Implements Eq. (2.5) of [\[Carrazza:2015dea\]][crate::bib::Carrazza2015dea].
pub fn gamma_phph(_c: &mut Cache, nf: u8) -> Complex<f64> {
    let cc = ChargeCombinations { nf };
    cmplx!(
        4.0 / 3.0 * (NC as f64) * ((cc.nu() as f64) * EU2 + (cc.nd() as f64) * ED2),
        0.
    )
}

/// Compute the non-singlet anomalous dimension.
///
/// Implements Eq. (2.5) of [\[Carrazza:2015dea\]][crate::bib::Carrazza2015dea].
pub fn gamma_ns(c: &mut Cache, nf: u8) -> Complex<f64> {
    as1::gamma_ns(c, nf) / CF
}

/// Compute the singlet anomalous dimension matrix.
pub fn gamma_singlet(c: &mut Cache, nf: u8) -> [[Complex<f64>; 4]; 4] {
    let cc = ChargeCombinations { nf };

    let gamma_ph_q = gamma_phq(c, nf);
    let gamma_q_ph = gamma_qph(c, nf);
    let gamma_nonsinglet = gamma_ns(c, nf);

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

/// Compute the valence anomalous dimension matrix.
pub fn gamma_valence(c: &mut Cache, nf: u8) -> [[Complex<f64>; 2]; 2] {
    let cc = ChargeCombinations { nf };
    let g = gamma_ns(c, nf);
    [
        [cc.e2avg() * g, cc.vue2m() * g],
        [cc.vde2m() * g, cc.e2delta() * g],
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
            let cc = ChargeCombinations { nf };
            assert_approx_eq_cmplx!(
                f64,
                EU2 * gamma_qph(&mut c, cc.nu())
                    + ED2 * gamma_qph(&mut c, cc.nd())
                    + gamma_phph(&mut c, nf),
                cmplx!(0., 0.),
                epsilon = 2e-6
            );
        }
    }
}
