//! Cache harmonic sums for given Mellin N.

use num::{complex::Complex, Zero};

use crate::harmonics::{g_functions, w1, w2, w3, w4, w5};

/// List of available elements.
#[derive(Debug, PartialEq, Eq)]
pub enum K {
    /// $S_1(N)$
    S1,
    /// $S_2(N)$
    S2,
    /// $S_3(N)$
    S3,
    /// $S_4(N)$
    S4,
    /// $S_5(N)$
    S5,
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
}

impl K {
    fn idx(&self) -> usize {
        match self {
            K::S1 => 0,
            K::S2 => 1,
            K::S3 => 2,
            K::S4 => 3,
            K::S5 => 4,
            K::S1h => 5,
            K::S2h => 6,
            K::S3h => 7,
            K::S1mh => 8,
            K::S2mh => 9,
            K::S3mh => 10,
            K::G3 => 11,
            K::Sm1e => 12,
            K::Sm1o => 13,
            K::Sm2e => 14,
            K::Sm2o => 15,
            K::Sm3e => 16,
            K::Sm3o => 17,
            K::Sm21e => 18,
            K::Sm21o => 19,
        }
    }
}

const CACHE_SIZE: usize = 20;

/// Hold all cached values.
pub struct Cache {
    /// Mellin N
    n: Complex<f64>,
    /// Flat lookup table indexed by K::idx()
    m: [Option<Complex<f64>>; CACHE_SIZE],
}

impl Cache {
    /// Initialize new, empty Cache at given Mellin N.
    pub fn new(n: Complex<f64>) -> Self {
        Self {
            n,
            m: [None; CACHE_SIZE],
        }
    }

    /// Get Mellin N.
    pub fn n(&self) -> Complex<f64> {
        Complex::new(self.n.re, self.n.im)
    }

    /// Retrieve an element.
    pub fn get(&mut self, k: K) -> Complex<f64> {
        let idx = k.idx();
        if let Some(val) = self.m[idx] {
            return val;
        }
        // compute new
        let val = match k {
            K::S1 => w1::S1(self.n),
            K::S2 => w2::S2(self.n),
            K::S3 => w3::S3(self.n),
            K::S4 => w4::S4(self.n),
            K::S5 => w5::S5(self.n),
            K::S1h => w1::S1(self.n / 2.),
            K::S2h => w2::S2(self.n / 2.),
            K::S3h => w3::S3(self.n / 2.),
            K::S1mh => w1::S1((self.n - 1.) / 2.),
            K::S2mh => w2::S2((self.n - 1.) / 2.),
            K::S3mh => w3::S3((self.n - 1.) / 2.),
            K::G3 => g_functions::g3(self),
            K::Sm1e => w1::Sm1e(self),
            K::Sm1o => w1::Sm1o(self),
            K::Sm2e => w2::Sm2e(self),
            K::Sm2o => w2::Sm2o(self),
            K::Sm3e => w3::Sm3e(self),
            K::Sm3o => w3::Sm3o(self),
            K::Sm21e => w3::Sm21e(self),
            K::Sm21o => w3::Sm21o(self),
        };
        // insert
        self.m[idx] = Some(val);
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
    fn test_cache_idx_mapping() {
        let all_variants = [
            K::S1,
            K::S2,
            K::S3,
            K::S4,
            K::S5,
            K::S1h,
            K::S2h,
            K::S3h,
            K::S1mh,
            K::S2mh,
            K::S3mh,
            K::G3,
            K::Sm1e,
            K::Sm1o,
            K::Sm2e,
            K::Sm2o,
            K::Sm3e,
            K::Sm3o,
            K::Sm21e,
            K::Sm21o,
        ];
        // size: number of variants matches CACHE_SIZE
        assert_eq!(all_variants.len(), CACHE_SIZE);

        let mut c = Cache::new(cmplx!(2., 0.));
        let mut seen = [false; CACHE_SIZE];
        for v in all_variants {
            let idx = v.idx();
            // mapping is continuous: no duplicate indices
            assert!(!seen[idx], "duplicate index {idx}");
            seen[idx] = true;
            // exercises Cache::get(), panics if idx() >= CACHE_SIZE (size check)
            c.get(v);
        }
        // every slot 0..CACHE_SIZE was covered (no holes)
        assert!(seen.iter().all(|&b| b), "index mapping has holes");
    }

    #[test]
    fn test_recursive_harmonic_sum() {
        const SX: [fn(Complex<f64>) -> Complex<f64>; 5] = [w1::S1, w2::S2, w3::S3, w4::S4, w5::S5];
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
