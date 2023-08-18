//! Cache harmonic sums for given Mellin N.

use num::complex::Complex;
use std::collections::HashMap;

use crate::harmonics::w1;

#[cfg_attr(doc, katexit::katexit)]
/// List of available elements.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum K {
    /// $S_1(N)$
    S1,
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
        };
        // insert
        self.m.insert(k, val);
        val
    }
}
