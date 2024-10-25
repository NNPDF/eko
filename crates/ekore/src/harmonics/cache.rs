//! Cache harmonic sums for given Mellin N.

use num::{complex::Complex, Zero};
use std::collections::HashMap;

use crate::harmonics::{g_functions, w1, w2, w3, w4};

/// List of available elements.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum K {
    /// $S_1(N)$
    S1,
    /// $S_2(N)$
    S2,
    /// $S_3(N)$
    S3,
    /// $S_4(N)$
    S4,
    /// $S_1(N/2)$
    S1h,
    /// $S_2(N/2)$
    S2h,
    /// $S_3(N/2)$
    S3h,
    /// $S_1((N-1)/2)$
    S1mh,
    /// $S_2((N-1)/2)$
    S2mh,
    /// $S_3((N-1)/2)$
    S3mh,
    /// $g_3(N)$
    G3,
    /// $S_{-1}(N)$ even moments
    Sm1e,
    /// $S_{-1}(N)$ odd moments
    Sm1o,
    /// $S_{-2}(N)$ even moments
    Sm2e,
    /// $S_{-2}(N)$ odd moments
    Sm2o,
    /// $S_{-3}(N)$ even moments
    Sm3e,
    /// $S_{-3}(N)$ odd moments
    Sm3o,
    /// $S_{-2,1}(N)$ even moments
    Sm21e,
    /// $S_{-2,1}(N)$ odd moments
    Sm21o,
    /// recursive harmonics
    S1ph,
    S2ph,
    S3ph,
    S1p2,
    G3p2,
}

/// Hold all cached values.
pub struct Cache {
    /// Mellin N
    n: Complex<f64>,
    /// Mapping
    m: HashMap<K, Complex<f64>>,
}

impl Cache {
    /// Initialize new, empty Cache at given Mellin N.
    pub fn new(n: Complex<f64>) -> Self {
        Self {
            n,
            m: HashMap::new(),
        }
    }

    /// Get Mellin N.
    pub fn n(&self) -> Complex<f64> {
        Complex::new(self.n.re, self.n.im)
    }

    /// Retrieve an element.
    pub fn get(&mut self, k: K) -> Complex<f64> {
        let val = self.m.get(&k);
        // already there?
        if let Some(value) = val {
            return *value;
        }
        // compute new
        let val = match k {
            K::S1 => w1::S1(self.n),
            K::S2 => w2::S2(self.n),
            K::S3 => w3::S3(self.n),
            K::S4 => w4::S4(self.n),
            K::S1h => w1::S1(self.n / 2.),
            K::S2h => w2::S2(self.n / 2.),
            K::S3h => w3::S3(self.n / 2.),
            K::S1mh => w1::S1((self.n - 1.) / 2.),
            K::S2mh => w2::S2((self.n - 1.) / 2.),
            K::S3mh => w3::S3((self.n - 1.) / 2.),
            K::G3 => g_functions::g3(self.n, self.get(K::S1)),
            K::G3p2 => g_functions::g3(self.n + 2., self.get(K::S1p2)),
            K::Sm1e => w1::Sm1e(self.get(K::S1), self.get(K::S1h)),
            K::Sm1o => w1::Sm1o(self.get(K::S1), self.get(K::S1mh)),
            K::Sm2e => w2::Sm2e(self.get(K::S2), self.get(K::S2h)),
            K::Sm2o => w2::Sm2o(self.get(K::S2), self.get(K::S2mh)),
            K::Sm3e => w3::Sm3e(self.get(K::S3), self.get(K::S3h)),
            K::Sm3o => w3::Sm3o(self.get(K::S3), self.get(K::S3mh)),
            K::Sm21e => w3::Sm21e(self.n, self.get(K::S1), self.get(K::Sm1e)),
            K::Sm21o => w3::Sm21o(self.n, self.get(K::S1), self.get(K::Sm1o)),
            K::S1ph => recursive_harmonic_sum(self.get(K::S1mh), (self.n - 1.) / 2., 1, 1),
            K::S2ph => recursive_harmonic_sum(self.get(K::S2mh), (self.n - 1.) / 2., 1, 2),
            K::S3ph => recursive_harmonic_sum(self.get(K::S3mh), (self.n - 1.) / 2., 1, 3),
            K::S1p2 => recursive_harmonic_sum(self.get(K::S1), self.n, 2, 1),
        };
        // insert
        self.m.insert(k, val);
        val
    }
}

/// Recursive computation of harmonic sums.
///
/// Compute the harmonic sum $S_{w}(N+k)$ stating from the value $S_{w}(N)$ via the recurrence relations.
pub fn recursive_harmonic_sum(
    base_value: Complex<f64>,
    n: Complex<f64>,
    iterations: usize,
    weight: u32,
) -> Complex<f64> {
    let mut fact = Complex::zero();
    for i in 1..iterations + 1 {
        fact += (1.0 / (n + (i as f64))).powu(weight);
    }
    base_value + fact
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmonics::{w1, w2, w3, w4};
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn n() {
        let n = cmplx!(1., 0.);
        let c = Cache::new(n);
        let mut m = c.n();
        m += cmplx!(1., 0.);
        assert_approx_eq_cmplx!(f64, c.n(), n);
        assert_approx_eq_cmplx!(f64, m, cmplx!(2., 0.));
    }

    #[test]
    fn test_recursive_harmonic_sum() {
        const SX: [fn(Complex<f64>) -> Complex<f64>; 4] = [w1::S1, w2::S2, w3::S3, w4::S4];
        const NS: [Complex<f64>; 2] = [cmplx!(1.0, 0.0), cmplx!(2.34, 3.45)];
        const ITERS: [usize; 2] = [1, 2];
        for sit in SX.iter().enumerate() {
            for nit in NS.iter().enumerate() {
                let n = *nit.1;
                for iit in ITERS.iter().enumerate() {
                    let iterations = *iit.1;
                    let s_base = sit.1(n);
                    let s_test = sit.1(n + (iterations as f64));
                    let s_ref = recursive_harmonic_sum(s_base, n, iterations, 1 + (sit.0 as u32));
                    assert_approx_eq_cmplx!(f64, s_test, s_ref);
                }
            }
        }
    }
}
