//! The unpolarized, space-like |OME|.
#![allow(non_snake_case)]

use ekore::operator_matrix_elements::unpolarized::spacelike;
use numpy::{Complex64, PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::cache::Cache;

/// Compute the tower of the singlet |OME|.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (supported range: < 3).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `L`: The logarithm parameter.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(matching_order_qcd, 3, 3)`.
#[pyfunction]
#[pyo3(signature = (matching_order_qcd, cache, nf, L))]
pub fn A_singlet<'py>(
    py: Python<'py>,
    matching_order_qcd: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
    L: f64,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    if matching_order_qcd >= 3 {
        return Err(PyValueError::new_err(format!(
            "matching_order_qcd must be < 3, got {matching_order_qcd}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let ome = spacelike::A_singlet(matching_order_qcd, &mut cache.inner, nf, L);

    let mut data: Vec<Complex64> = Vec::with_capacity(matching_order_qcd * 9);
    for mat in ome.into_iter().take(matching_order_qcd) {
        for row in mat.iter() {
            for v in row.iter() {
                data.push(Complex64::new(v.re, v.im));
            }
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([matching_order_qcd, 3, 3])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the tower of the non-singlet |OME|.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (supported range: < 3).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `L`: The logarithm parameter.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(matching_order_qcd, 2, 2)`.
#[pyfunction]
#[pyo3(signature = (matching_order_qcd, cache, nf, L))]
pub fn A_non_singlet<'py>(
    py: Python<'py>,
    matching_order_qcd: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
    L: f64,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    if matching_order_qcd >= 3 {
        return Err(PyValueError::new_err(format!(
            "matching_order_qcd must be < 3, got {matching_order_qcd}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let ome = spacelike::A_non_singlet(matching_order_qcd, &mut cache.inner, nf, L);

    let mut data: Vec<Complex64> = Vec::with_capacity(matching_order_qcd * 4);
    for mat in ome.into_iter().take(matching_order_qcd) {
        for row in mat.iter() {
            for v in row.iter() {
                data.push(Complex64::new(v.re, v.im));
            }
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([matching_order_qcd, 2, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(A_singlet, m)?)?;
    m.add_function(wrap_pyfunction!(A_non_singlet, m)?)?;
    Ok(())
}
