//! The polarized, space-like anomalous dimensions.

use ekore::anomalous_dimensions::polarized::spacelike;
use numpy::{Complex64, PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::cache::Cache;

/// Compute the tower of non-singlet |QCD| anomalous dimensions.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (supported range: < 3).
/// * `mode`: The specific non-singlet sector.
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd,)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, mode, cache, nf))]
pub fn gamma_ns_qcd<'py>(
    py: Python<'py>,
    order_qcd: usize,
    mode: u16,
    cache: &Bound<'py, Cache>,
    nf: u8,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    gamma_ns_qcd_body!(
        py,
        order_qcd,
        mode,
        cache,
        nf,
        order_qcd >= 3,
        spacelike::gamma_ns_qcd
    )
}

/// Compute the tower of singlet |QCD| anomalous dimension matrices.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (supported range: < 3).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd, 2, 2)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, cache, nf))]
pub fn gamma_singlet_qcd<'py>(
    py: Python<'py>,
    order_qcd: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    gamma_singlet_qcd_body!(
        py,
        order_qcd,
        cache,
        nf,
        order_qcd >= 3,
        spacelike::gamma_singlet_qcd,
        2
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamma_ns_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qcd, m)?)?;
    Ok(())
}
