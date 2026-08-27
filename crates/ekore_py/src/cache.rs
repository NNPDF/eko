//! Python bindings for the Mellin-space harmonics cache.

use ekore::harmonics::cache::Cache as EkoreCache;
use numpy::Complex64;
use pyo3::prelude::*;

/// Mellin-space harmonics cache.
///
/// Memoizes harmonic sums so repeated evaluations at the same Mellin N don't recompute them.
///
/// # Parameters
/// * `n` : Mellin variable N.
///
/// # Returns
/// * Returns a `Cache` at the given parameter.
#[pyclass(name = "Cache", module = "ekore_rs")]
pub struct Cache {
    pub(crate) inner: EkoreCache,
    n: Complex64,
}

#[pymethods]
impl Cache {
    #[new]
    fn new(n: Complex64) -> Self {
        Self {
            inner: EkoreCache::new(n),
            n,
        }
    }

    #[getter]
    fn n(&self) -> Complex64 {
        self.n
    }

    fn __repr__(&self) -> String {
        format!("Cache(n=({}{:+}j))", self.n.re, self.n.im)
    }
}
