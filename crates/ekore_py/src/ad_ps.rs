//! The polarized, space-like anomalous dimensions.

use ekore::anomalous_dimensions::polarized::spacelike;
use ekore::constants::{PID_NSM, PID_NSP, PID_NSV};
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
    if order_qcd >= 3 {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be < 3, got {order_qcd}"
        )));
    }

    if !matches!(mode, PID_NSP | PID_NSM | PID_NSV) {
        return Err(PyValueError::new_err(format!(
            "invalid non-singlet mode: {mode}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma = spacelike::gamma_ns_qcd(order_qcd, mode, &mut cache.inner, nf);

    let data: Vec<Complex64> = gamma
        .into_iter()
        .take(order_qcd)
        .map(|c| Complex64::new(c.re, c.im))
        .collect();
    Ok(PyArray1::from_vec(py, data))
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
    if order_qcd >= 3 {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be < 3, got {order_qcd}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma = spacelike::gamma_singlet_qcd(order_qcd, &mut cache.inner, nf);

    let mut data: Vec<Complex64> = Vec::with_capacity(order_qcd * 4);
    for mat in gamma.into_iter().take(order_qcd) {
        for row in mat.iter() {
            for v in row.iter() {
                data.push(Complex64::new(v.re, v.im));
            }
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([order_qcd, 2, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamma_ns_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qcd, m)?)?;
    Ok(())
}
