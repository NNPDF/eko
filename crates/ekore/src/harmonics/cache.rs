//! Cache harmonic sums for given Mellin N.

use hashbrown::HashMap;
use num::{complex::Complex, Zero};

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
}

/// Hold all cached values.
pub struct Cache {
    /// Mellin N
    pub n: Complex<f64>,
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
    use crate::harmonics::cache::recursive_harmonic_sum;
    use crate::harmonics::{w1, w2, w3, w4};
    use crate::{assert_approx_eq_cmplx, cmplx};
    use num::complex::Complex;

    #[test]
    fn test_recursive_harmonic_sum() {
        const SX: [fn(Complex<f64>) -> Complex<f64>; 4] = [w1::S1, w2::S2, w3::S3, w4::S4];
        const NS: [Complex<f64>; 2] = [cmplx![1.0, 0.0], cmplx![2.34, 3.45]];
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
