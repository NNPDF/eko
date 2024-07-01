//! NLO QCD

use num::complex::Complex;
use num::Zero;

use crate::constants::CF;
use crate::harmonics::cache::{Cache, K};

/// Compute heavy-heavy |OME| :math:`A_{HH}^{(1)}`
///
/// Defined in Eq. () of .
pub fn A_hh(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n;
    let S1m = c.get(K::S1) - 1. / N;
    let S2m = c.get(K::S2) - 1. / N.powu(2);
    let ahh_l = (2. + N - 3. * N.powu(2)) / (N * (1. + N)) + 4. * S1m;
    let ahh = 2.
        * (2. + 5. * N + N.powu(2)
            - 6. * N.powu(3)
            - 2. * N.powu(4)
            - 2. * N * (-1. - 2. * N + N.powu(3)) * S1m)
        / (N * (1. + N)).powu(2)
        + 4. * (S1m.powu(2) + S2m);

    -CF * (ahh_l * L + ahh)
}

/// Compute gluon-heavy |OME| :math:`A_{gH}^{(1)}`
///
/// Defined in Eq. () of .
pub fn A_gh(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n;
    let agh_l1 = (2. + N + N.powu(2)) / (N * (N.powu(2) - 1.));
    let agh_l0 = (-4. + 2. * N + N.powu(2) * (15. + N * (3. + N - N.powu(2))))
        / (N * (N.powu(2) - 1.)).powu(2);
    2. * CF * (agh_l0 + agh_l1 * L)
}

/// Compute heavy-gluon |OME| :math:`A_{Hg}^{(1)}`
///
/// Defined in Eq. () of .
pub fn A_hg(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n;
    let den = 1. / (N * (N + 1.) * (2. + N));
    let num = 2. * (2. + N + N.powu(2));
    num * den * L
}

/// Compute gluon-gluon |OME| :math:`A_{gg}^{(1)}`
///
/// Defined in Eq. () of .
pub fn A_gg(_c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    (-2.0 / 3.0 * L).into()
}

/// Compute the |NLO| singlet |OME|
pub fn A_singlet(c: &mut Cache, _nf: u8, L: f64) -> [[Complex<f64>; 3]; 3] {
    [
        [A_gg(c, _nf, L), Complex::<f64>::zero(), A_gh(c, _nf, L)],
        [
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
            Complex::<f64>::zero(),
        ],
        [A_hg(c, _nf, L), Complex::<f64>::zero(), A_hh(c, _nf, L)],
    ]
}

/// Compute the |NLO| non singlet |OME|
pub fn A_ns(c: &mut Cache, _nf: u8, L: f64) -> [[Complex<f64>; 2]; 2] {
    [
        [Complex::<f64>::zero(), Complex::<f64>::zero()],
        [Complex::<f64>::zero(), A_hh(c, _nf, L)],
    ]
}

#[cfg(test)]
mod tests {
    use crate::cmplx;
    use crate::{
        harmonics::cache::Cache, operator_matrix_elements::unpolarized::spacelike::as1::*,
    };
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;
    const NF: u8 = 5;

    #[test]
    fn test_momentum_conservation() {
        const N: Complex<f64> = cmplx![2., 0.];
        const L: f64 = 100.;
        let mut c = Cache::new(N);
        let aS1 = A_singlet(&mut c, NF, L);
        // heavy quark momentum conservation
        assert_approx_eq!(
            f64,
            (aS1[0][2] + aS1[1][2] + aS1[2][2]).re,
            0.,
            epsilon = 1e-10
        );
        assert_approx_eq!(
            f64,
            (aS1[0][2] + aS1[1][2] + aS1[2][2]).im,
            0.,
            epsilon = 1e-10
        );
        // gluon momentum conservation
        assert_approx_eq!(f64, (aS1[0][0] + aS1[1][0] + aS1[2][0]).re, 0.);
        assert_approx_eq!(f64, (aS1[0][0] + aS1[1][0] + aS1[2][0]).im, 0.);
    }

    #[test]
    fn test_A1_intrinsic() {
        const N: Complex<f64> = cmplx![2., 0.];
        const L: f64 = 3.0;
        let mut c = Cache::new(N);
        let aNS1i = A_ns(&mut c, NF, L);
        let aS1i = A_singlet(&mut c, NF, L);
        assert_eq!(aNS1i[1][1], aS1i[2][2]);
    }

    #[test]
    /// Test against Blumlein OME implementation Ref.()
    /// Only even moments are available in that code.
    fn test_Blumlein_1() {
        const L: f64 = 10.;
        let ref_val_gg = [-6.66667, -6.66667, -6.66667, -6.66667, -6.66667];
        let ref_val_Hg = [6.66667, 3.66667, 2.61905, 2.05556, 1.69697];

        for n in 0..4 {
            let N = cmplx![2. * (n as f64) + 2., 0.];
            let mut c = Cache::new(N);
            let aS1 = A_singlet(&mut c, NF, L);
            // lower numerical accuracy than python?
            assert_approx_eq!(f64, aS1[0][0].re, ref_val_gg[n], epsilon = 4e-6);
            assert_approx_eq!(f64, aS1[2][0].re, ref_val_Hg[n], epsilon = 5e-6);
        }
    }
}
