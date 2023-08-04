use num::complex::Complex;
use std::collections::HashMap;

use crate::harmonics::w1;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum K {
    S1,
}

pub struct Cache {
    pub n: Complex<f64>,
    m: HashMap<K, Complex<f64>>,
}

impl Cache {
    pub fn get(&mut self, k: K) -> Complex<f64> {
        let val = self.m.get(&k);
        if val.is_some() {
            return *val.unwrap();
        }
        let val = match k {
            K::S1 => w1::S1(self.n),
        };
        self.m.insert(k, val);
        val
    }
}
