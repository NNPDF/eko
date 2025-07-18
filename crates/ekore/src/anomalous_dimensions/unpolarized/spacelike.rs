//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::constants::{
    ED2, EU2, PID_NSM, PID_NSM_D, PID_NSM_U, PID_NSP, PID_NSP_D, PID_NSP_U, PID_NSV,
};
use crate::harmonics::cache::Cache;
use num::complex::Complex;
use num::Zero;
pub mod aem1;
pub mod aem2;
pub mod as1;
pub mod as1aem1;
pub mod as2;
pub mod as3;
pub mod as4;

/// Compute the tower of the non-singlet anomalous dimensions.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    c: &mut Cache,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> Vec<Complex<f64>> {
    let mut gamma_ns = vec![Complex::<f64>::zero(); order_qcd];
    gamma_ns[0] = as1::gamma_ns(c, nf);
    // NLO and beyond
    if order_qcd >= 2 {
        let gamma_ns_1 = match mode {
            PID_NSP => as2::gamma_nsp(c, nf),
            // To fill the full valence vector in NNLO we need to add gamma_ns^1 explicitly here
            PID_NSM | PID_NSV => as2::gamma_nsm(c, nf),
            _ => panic!("Unknown non-singlet sector element"),
        };
        gamma_ns[1] = gamma_ns_1
    }
    // NNLO and beyond
    if order_qcd >= 3 {
        let gamma_ns_2 = match mode {
            PID_NSP => as3::gamma_nsp(c, nf),
            PID_NSM => as3::gamma_nsm(c, nf),
            PID_NSV => as3::gamma_nsv(c, nf),
            _ => panic!("Unknown non-singlet sector element"),
        };
        gamma_ns[2] = gamma_ns_2
    }
    // N3LO and beyond
    if order_qcd >= 4 {
        let gamma_ns_3 = match mode {
            PID_NSP => as4::gamma_nsp(c, nf, n3lo_variation[0]),
            PID_NSM => as4::gamma_nsm(c, nf, n3lo_variation[1]),
            PID_NSV => as4::gamma_nsv(c, nf, n3lo_variation[2]),
            _ => panic!("Unknown non-singlet sector element"),
        };
        gamma_ns[3] = gamma_ns_3
    }
    gamma_ns
}

/// Compute the tower of the singlet anomalous dimension matrices.
///
/// `n3lo_variation = (gg, gq, qg, qq)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_singlet_qcd(
    order_qcd: usize,
    c: &mut Cache,
    nf: u8,
    n3lo_variation: [u8; 4],
) -> Vec<[[Complex<f64>; 2]; 2]> {
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
    // N3LO and beyond
    if order_qcd >= 4 {
        gamma_S[3] = as4::gamma_singlet(c, nf, n3lo_variation);
    }
    gamma_S
}

/// Compute the tower of the |QCD| x |QED| non-singlet anomalous dimensions.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_ns_qed(
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    c: &mut Cache,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> Vec<Vec<Complex<f64>>> {
    let col = vec![Complex::<f64>::zero(); order_qed + 1];
    let mut gamma_ns = vec![col; order_qcd + 1];

    // QCD corrections
    let qcd_mode = match mode {
        PID_NSP_U | PID_NSP_D => PID_NSP,
        PID_NSM_U | PID_NSM_D => PID_NSM,
        _ => mode,
    };
    let gamma_qcd = gamma_ns_qcd(order_qcd, qcd_mode, c, nf, n3lo_variation);
    for j in 0..order_qcd {
        gamma_ns[1 + j][0] = gamma_qcd[j];
    }
    // QED corrections
    gamma_ns[0][1] = match mode {
        PID_NSP_U | PID_NSM_U => EU2 * aem1::gamma_ns(c, nf),
        PID_NSP_D | PID_NSM_D => ED2 * aem1::gamma_ns(c, nf),
        _ => panic!("Unknown non-singlet sector element"),
    };
    if order_qed >= 2 {
        gamma_ns[0][2] = match mode {
            PID_NSP_U => EU2 * aem2::gamma_nspu(c, nf),
            PID_NSP_D => ED2 * aem2::gamma_nspd(c, nf),
            PID_NSM_U => EU2 * aem2::gamma_nsmu(c, nf),
            PID_NSM_D => ED2 * aem2::gamma_nsmd(c, nf),
            _ => panic!("Unknown non-singlet sector element"),
        };
    }
    // QCDxQED corrections
    gamma_ns[1][1] = match mode {
        PID_NSP_U => EU2 * as1aem1::gamma_nsp(c, nf),
        PID_NSP_D => ED2 * as1aem1::gamma_nsp(c, nf),
        PID_NSM_U => EU2 * as1aem1::gamma_nsm(c, nf),
        PID_NSM_D => ED2 * as1aem1::gamma_nsm(c, nf),
        _ => panic!("Unknown non-singlet sector element"),
    };
    gamma_ns
}

/// Compute the tower of the |QCD| x |QED| singlet anomalous dimensions matrices.
///
/// `n3lo_variation = (gg, gq, qg, qq, ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_singlet_qed(
    order_qcd: usize,
    order_qed: usize,
    c: &mut Cache,
    nf: u8,
    n3lo_variation: [u8; 7],
) -> Vec<Vec<[[Complex<f64>; 4]; 4]>> {
    let col = vec![
        [[
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero()
        ]; 4];
        order_qed + 1
    ];
    let mut gamma_s = vec![col; order_qcd + 1];

    // QCD corrections
    let gamma_qcd_s = gamma_singlet_qcd(order_qcd, c, nf, n3lo_variation[0..4].try_into().unwrap());
    let gamma_qcd_nsp = gamma_ns_qcd(
        order_qcd,
        PID_NSP,
        c,
        nf,
        n3lo_variation[4..7].try_into().unwrap(),
    );
    for j in 0..order_qcd {
        let s = gamma_qcd_s[j];
        gamma_s[1 + j][0] = [
            [
                s[1][1],
                Complex::<f64>::zero(),
                s[1][0],
                Complex::<f64>::zero(),
            ],
            [
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
            ],
            [
                s[0][1],
                Complex::<f64>::zero(),
                s[0][0],
                Complex::<f64>::zero(),
            ],
            [
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                Complex::<f64>::zero(),
                gamma_qcd_nsp[j],
            ],
        ];
    }
    // QED corrections
    gamma_s[0][1] = aem1::gamma_singlet(c, nf);
    if order_qed >= 2 {
        gamma_s[0][2] = aem2::gamma_singlet(c, nf);
    }
    // QCDxQED corrections
    gamma_s[1][1] = as1aem1::gamma_singlet(c, nf);
    gamma_s
}

/// Compute the tower of the |QCD| x |QED| valence anomalous dimensions matrices.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
pub fn gamma_valence_qed(
    order_qcd: usize,
    order_qed: usize,
    c: &mut Cache,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> Vec<Vec<[[Complex<f64>; 2]; 2]>> {
    let col = vec![[[Complex::<f64>::zero(), Complex::<f64>::zero(),]; 2]; order_qed + 1];
    let mut gamma_v = vec![col; order_qcd + 1];

    // QCD corrections
    let gamma_qcd_nsv = gamma_ns_qcd(order_qcd, PID_NSV, c, nf, n3lo_variation);
    let gamma_qcd_nsm = gamma_ns_qcd(order_qcd, PID_NSM, c, nf, n3lo_variation);
    for j in 0..order_qcd {
        gamma_v[1 + j][0] = [
            [gamma_qcd_nsv[j], Complex::<f64>::zero()],
            [Complex::<f64>::zero(), gamma_qcd_nsm[j]],
        ];
    }
    // QED corrections
    gamma_v[0][1] = aem1::gamma_valence(c, nf);
    if order_qed >= 2 {
        gamma_v[0][2] = aem2::gamma_valence(c, nf);
    }
    // QCDxQED corrections
    gamma_v[1][1] = as1aem1::gamma_valence(c, nf);
    gamma_v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_approx_eq_cmplx, assert_approx_eq_cmplx_1d, assert_approx_eq_cmplx_2d, cmplx,
    };
    use num::complex::Complex;

    #[test]
    fn test_gamma_ns_qcd() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let n3lo_variation: [u8; 3] = [0, 0, 0];
        assert_approx_eq_cmplx!(
            f64,
            gamma_ns_qcd(3, PID_NSP, &mut c, NF, n3lo_variation)[0],
            cmplx!(0., 0.),
            epsilon = 1e-14
        );

        assert_approx_eq_cmplx_1d!(
            f64,
            gamma_ns_qcd(2, PID_NSM, &mut c, NF, n3lo_variation),
            [cmplx!(0., 0.); 2],
            2,
            epsilon = 2e-6
        );

        assert_approx_eq_cmplx_1d!(
            f64,
            gamma_ns_qcd(3, PID_NSM, &mut c, NF, n3lo_variation),
            [cmplx!(0., 0.); 3],
            3,
            epsilon = 2e-4
        );

        assert_approx_eq_cmplx_1d!(
            f64,
            gamma_ns_qcd(3, PID_NSV, &mut c, NF, n3lo_variation),
            [cmplx!(0., 0.); 3],
            3,
            epsilon = 8e-4
        );
    }

    #[test]
    fn test_gamma_ns_qed() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(1., 0.);
        let mut c = Cache::new(N);
        let n3lo_variation: [u8; 3] = [0, 0, 0];

        // ns-
        for pid in [PID_NSM_U, PID_NSM_D] {
            assert_approx_eq_cmplx_2d!(
                f64,
                gamma_ns_qed(1, 1, pid, &mut c, NF, n3lo_variation),
                [[cmplx!(0., 0.); 2]; 2],
                2,
                epsilon = 1e-5
            );
        }

        // ns+
        for pid in [PID_NSP_U, PID_NSP_U] {
            // as^0 a^0 must be trivial
            assert_approx_eq_cmplx!(
                f64,
                gamma_ns_qed(1, 1, pid, &mut c, NF, n3lo_variation)[0][0],
                cmplx!(0., 0.)
            );
            assert_approx_eq_cmplx!(
                f64,
                gamma_ns_qed(1, 1, pid, &mut c, NF, n3lo_variation)[0][1],
                cmplx!(0., 0.),
                epsilon = 1e-5
            );
        }
    }

    #[test]
    fn test_gamma_valence_qed() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let n3lo_variation: [u8; 3] = [0, 0, 0];

        let g = gamma_valence_qed(3, 2, &mut c, NF, n3lo_variation);
        // as^0 a^0 must be trivial
        assert_approx_eq_cmplx_2d!(f64, g[0][0], [[cmplx!(0., 0.); 2]; 2], 2);
        // reference from Python side
        assert_approx_eq_cmplx_2d!(
            f64,
            g[3][0],
            [
                [cmplx!(459.646893789751, 0.), cmplx!(0., 0.)],
                [cmplx!(0., 0.), cmplx!(437.60340375, 0.)]
            ],
            2,
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_gamma_singlet_qed() {
        const NF: u8 = 3;
        const N: Complex<f64> = cmplx!(2., 0.);
        let mut c = Cache::new(N);
        let n3lo_variation: [u8; 7] = [0, 0, 0, 0, 0, 0, 0];
        let g = gamma_singlet_qed(3, 2, &mut c, NF, n3lo_variation);
        // as^0 a^0 must be trivial
        assert_approx_eq_cmplx_2d!(f64, g[0][0], [[cmplx!(0., 0.); 4]; 4], 4);

        // reference from Python side
        assert_approx_eq_cmplx_2d!(
            f64,
            g[3][0],
            [
                [
                    cmplx!(3.857918949669738, 0.),
                    cmplx!(0., 0.),
                    cmplx!(-290.42193908689745, 0.),
                    cmplx!(0., 0.)
                ],
                [
                    cmplx!(0., 0.),
                    cmplx!(0., 0.),
                    cmplx!(0., 0.),
                    cmplx!(0., 0.)
                ],
                [
                    cmplx!(-3.859554320251334, 0.),
                    cmplx!(0., 0.),
                    cmplx!(290.4252052962147, 0.),
                    cmplx!(0., 0.)
                ],
                [
                    cmplx!(0., 0.),
                    cmplx!(0., 0.),
                    cmplx!(0., 0.),
                    cmplx!(448.0752570151872, 0.)
                ]
            ],
            4,
            epsilon = 1e-5
        );
    }
}
