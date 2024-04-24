//! Cache harmonic sums for given Mellin N.

use hashbrown::HashMap;
use num::complex::Complex;

use crate::harmonics::{w1, w2, w3};

#[cfg_attr(doc, katexit::katexit)]
/// List of available elements.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum K {
    /// $S_1(N)$
    S1,
    /// $S_2(N)$
    S2,
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
        if val.is_some() {
            return *val.unwrap();
        }
        // compute new
        let val = match k {
            K::S1 => w1::S1(self.n),
            K::S2 => w2::S2(self.n),
            K::S1h => w1::S1(self.n / 2.),
            K::S2h => w2::S2(self.n / 2.),
            K::S3h => w3::S3(self.n / 2.),
            K::S1mh => w1::S1((self.n - 1.) / 2.),
            K::S2mh => w2::S2((self.n - 1.) / 2.),
            K::S3mh => w3::S3((self.n - 1.) / 2.),
        };
        // insert
        self.m.insert(k, val);
        val
    }
}
