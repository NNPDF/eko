//! The unpolarized, space-like anomalous dimensions.

use ekore::anomalous_dimensions::unpolarized::spacelike;
use ekore::constants::{MAX_ORDER_QCD, MAX_ORDER_QED};
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
    gamma_ns_qcd_body!(
        py,
        order_qcd,
        mode,
        cache,
        nf,
        n3lo_variation,
        order_qcd > MAX_ORDER_QCD,
        spacelike::gamma_ns_qcd
    )
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
    gamma_singlet_qcd_body!(
        py,
        order_qcd,
        cache,
        nf,
        n3lo_variation,
        order_qcd > MAX_ORDER_QCD,
        spacelike::gamma_singlet_qcd,
        2
    )
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
    gamma_ns_qed_body!(
        py,
        order_qcd,
        order_qed,
        mode,
        cache,
        nf,
        n3lo_variation,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_ns_qed
    )
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
    gamma_qed_matrix_body!(
        py,
        order_qcd,
        order_qed,
        cache,
        nf,
        n3lo_variation,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_singlet_qed,
        4
    )
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
    gamma_qed_matrix_body!(
        py,
        order_qcd,
        order_qed,
        cache,
        nf,
        n3lo_variation,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_valence_qed,
        2
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gamma_ns_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qcd, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_ns_qed, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_singlet_qed, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_valence_qed, m)?)?;
    Ok(())
}
