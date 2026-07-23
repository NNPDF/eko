//! The unpolarized, space-like anomalous dimensions.

use ekore::anomalous_dimensions::unpolarized::spacelike;
use ekore::constants::{
    MAX_ORDER_QCD, MAX_ORDER_QED, PID_NSM, PID_NSM_D, PID_NSM_U, PID_NSP, PID_NSP_D, PID_NSP_U,
    PID_NSV,
};
use numpy::{Complex64, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::cache::Cache;

/// Compute the tower of the non-singlet |QCD| anomalous dimensions.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `mode`: The specific non-singlet sector.
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: The three N3LO variation flags.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd,)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, mode, cache, nf, n3lo_variation))]
pub fn gamma_ns_qcd<'py>(
    py: Python<'py>,
    order_qcd: usize,
    mode: u16,
    cache: &Bound<'py, Cache>,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    if order_qcd > MAX_ORDER_QCD {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be <= {MAX_ORDER_QCD}, got {order_qcd}"
        )));
    }

    if !matches!(mode, PID_NSP | PID_NSM | PID_NSV) {
        return Err(PyValueError::new_err(format!(
            "invalid non-singlet mode: {mode}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma = spacelike::gamma_ns_qcd(order_qcd, mode, &mut cache.inner, nf, n3lo_variation);

    let data: Vec<Complex64> = gamma
        .into_iter()
        .take(order_qcd)
        .map(|c| Complex64::new(c.re, c.im))
        .collect();
    Ok(PyArray1::from_vec(py, data))
}

/// Compute the tower of the singlet |QCD| anomalous dimension matrices.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: The four N3LO variation flags.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd, 2, 2)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, cache, nf, n3lo_variation))]
pub fn gamma_singlet_qcd<'py>(
    py: Python<'py>,
    order_qcd: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
    n3lo_variation: [u8; 4],
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    if order_qcd > MAX_ORDER_QCD {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be <= {MAX_ORDER_QCD}, got {order_qcd}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma = spacelike::gamma_singlet_qcd(order_qcd, &mut cache.inner, nf, n3lo_variation);

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

/// Compute the tower of the |QCD| x |QED| non-singlet anomalous dimensions.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `mode`: The specific non-singlet sector.
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: The three N3LO variation flags.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd + 1, order_qed + 1)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, order_qed, mode, cache, nf, n3lo_variation))]
pub fn gamma_ns_qed<'py>(
    py: Python<'py>,
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    cache: &Bound<'py, Cache>,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be <= {MAX_ORDER_QCD} and order_qed must be <= {MAX_ORDER_QED}, got {order_qcd}, {order_qed}"
        )));
    }

    if !matches!(
        mode,
        PID_NSP_U | PID_NSP_D | PID_NSM_U | PID_NSM_D | PID_NSP | PID_NSM | PID_NSV
    ) {
        return Err(PyValueError::new_err(format!(
            "invalid non-singlet mode: {mode}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma = spacelike::gamma_ns_qed(
        order_qcd,
        order_qed,
        mode,
        &mut cache.inner,
        nf,
        n3lo_variation,
    );

    let mut data: Vec<Complex64> = Vec::with_capacity((order_qcd + 1) * (order_qed + 1));
    for row in gamma.into_iter().take(order_qcd + 1) {
        for v in row.into_iter().take(order_qed + 1) {
            data.push(Complex64::new(v.re, v.im));
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([order_qcd + 1, order_qed + 1])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the tower of the |QCD| x |QED| singlet anomalous dimension matrices.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: The seven N3LO variation flags.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd + 1, order_qed + 1, 4, 4)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, order_qed, cache, nf, n3lo_variation))]
pub fn gamma_singlet_qed<'py>(
    py: Python<'py>,
    order_qcd: usize,
    order_qed: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
    n3lo_variation: [u8; 7],
) -> PyResult<Bound<'py, PyArray4<Complex64>>> {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be <= {MAX_ORDER_QCD} and order_qed must be <= {MAX_ORDER_QED}, got {order_qcd}, {order_qed}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma =
        spacelike::gamma_singlet_qed(order_qcd, order_qed, &mut cache.inner, nf, n3lo_variation);

    let mut data: Vec<Complex64> = Vec::with_capacity((order_qcd + 1) * (order_qed + 1) * 16);
    for row in gamma.into_iter().take(order_qcd + 1) {
        for mat in row.into_iter().take(order_qed + 1) {
            for r in mat.iter() {
                for v in r.iter() {
                    data.push(Complex64::new(v.re, v.im));
                }
            }
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([order_qcd + 1, order_qed + 1, 4, 4])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the tower of the |QCD| x |QED| valence anomalous dimension matrices.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `cache`: Harmonic sums cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: The three N3LO variation flags.
///
/// # Returns
/// A `numpy.ndarray`, complex array of shape `(order_qcd + 1, order_qed + 1, 2, 2)`.
#[pyfunction]
#[pyo3(signature = (order_qcd, order_qed, cache, nf, n3lo_variation))]
pub fn gamma_valence_qed<'py>(
    py: Python<'py>,
    order_qcd: usize,
    order_qed: usize,
    cache: &Bound<'py, Cache>,
    nf: u8,
    n3lo_variation: [u8; 3],
) -> PyResult<Bound<'py, PyArray4<Complex64>>> {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return Err(PyValueError::new_err(format!(
            "order_qcd must be <= {MAX_ORDER_QCD} and order_qed must be <= {MAX_ORDER_QED}, got {order_qcd}, {order_qed}"
        )));
    }

    let mut cache = cache.borrow_mut();
    let gamma =
        spacelike::gamma_valence_qed(order_qcd, order_qed, &mut cache.inner, nf, n3lo_variation);

    let mut data: Vec<Complex64> = Vec::with_capacity((order_qcd + 1) * (order_qed + 1) * 4);
    for row in gamma.into_iter().take(order_qcd + 1) {
        for mat in row.into_iter().take(order_qed + 1) {
            for r in mat.iter() {
                for v in r.iter() {
                    data.push(Complex64::new(v.re, v.im));
                }
            }
        }
    }

    PyArray1::from_vec(py, data)
        .reshape([order_qcd + 1, order_qed + 1, 2, 2])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamma_ns_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_ns_qed, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qed, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_valence_qed, m)?)?;
    Ok(())
}
