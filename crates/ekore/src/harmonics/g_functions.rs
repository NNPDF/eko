//! Auxilary functions for harmonics sums of weight = 3,4.

use crate::constants::ZETA2;
use crate::harmonics::cache::recursive_harmonic_sum as s;
use crate::harmonics::cache::{Cache, K};
use num::{complex::Complex, Zero};

/// Compute the Mellin transform of $\text{Li}_2(x)/(1+x)$.
///
/// This function appears in the analytic continuation of the harmonic sum
/// $S_{-2,1}(N)$ which in turn appears in the NLO anomalous dimension.
///
/// We use the name from [\[MuselliPhD\]](crate::bib::MuselliPhD), but not his implementation - rather we use the
/// Pegasus [\[Vogt:2004ns\]](crate::bib::Vogt2004ns) implementation.
pub fn g3(c: &mut Cache) -> Complex<f64> {
    let N = c.n();
    let S1 = c.get(K::S1);
    const CS: [f64; 7] = [
        1.0000e0, -0.9992e0, 0.9851e0, -0.9005e0, 0.6621e0, -0.3174e0, 0.0699e0,
    ];
    let mut g3 = Complex::zero();
    for cit in CS.iter().enumerate() {
        let Nj = N + (cit.0 as f64);
        g3 += (*cit.1) * (ZETA2 - s(S1, N, cit.0, 1) / Nj) / Nj;
    }
    g3
}

#[cfg(test)]
mod tests {
    use crate::harmonics::cache::Cache;
    use crate::harmonics::g_functions::g3;
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_mellin_g3() {
        const NS: [Complex<f64>; 3] = [cmplx!(1.0, 0.0), cmplx!(2.0, 0.0), cmplx!(1.0, 1.0)];
        // NIntegrate[x^({1, 2, 1 + I} - 1) PolyLog[2, x]/(1 + x), {x, 0, 1}]
        const REFVALS: [Complex<f64>; 3] = [
            cmplx!(0.3888958462, 0.),
            cmplx!(0.2560382207, 0.),
            cmplx!(0.3049381491, -0.1589060625),
        ];
        for it in NS.iter().enumerate() {
            let n = *it.1;
            let mut c = Cache::new(n);
            let refval = REFVALS[it.0];
            let g3 = g3(&mut c);
            assert_approx_eq_cmplx!(f64, g3, refval, epsilon = 1e-6);
        }
    }
}
