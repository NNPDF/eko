//! Interface to the eko Python package.

use ekore::anomalous_dimensions::polarized::spacelike as ad_ps;
use ekore::anomalous_dimensions::unpolarized::spacelike as ad_us;
// use ekore::anomalous_dimensions::unpolarized::timelike as ad_ut; TODO
use ekore::harmonics::cache::Cache;
// use ekore::operator_matrix_elements::polarized::spacelike as ome_ps; TODO
use ekore::operator_matrix_elements::unpolarized::spacelike as ome_us;
// use ekore::operator_matrix_elements::unpolarized::timelike as ome_ut; TODO
use num::Complex;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

pub mod bib;
pub mod mellin;

/// Wrapper to pass arguments back to Python.
struct RawCmplx {
    re: Vec<f64>,
    im: Vec<f64>,
}

/// Map tensor with shape (o,d,d) to c-ordered list.
///
/// This is needed for the QCD singlet.
fn unravel<const DIM: usize>(res: Vec<[[Complex<f64>; DIM]; DIM]>, order_qcd: usize) -> RawCmplx {
    let mut target = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };
    for obj in res.iter().take(order_qcd) {
        for col in obj.iter().take(DIM) {
            for el in col.iter().take(DIM) {
                target.re.push(el.re);
                target.im.push(el.im);
            }
        }
    }
    target
}

/// Map tensor with shape (o,o',d,d) to c-ordered list.
///
/// This is needed for the QED singlet and valence.
fn unravel_qed<const DIM: usize>(
    res: Vec<Vec<[[Complex<f64>; DIM]; DIM]>>,
    order_qcd: usize,
    order_qed: usize,
) -> RawCmplx {
    let mut target = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };
    for obj_ in res.iter().take(order_qcd + 1) {
        for obj in obj_.iter().take(order_qed + 1) {
            for col in obj.iter().take(DIM) {
                for el in col.iter().take(DIM) {
                    target.re.push(el.re);
                    target.im.push(el.im);
                }
            }
        }
    }
    target
}

/// Map tensor with shape (o,o',d) to c-ordered list.
///
/// This is needed for the QED non-singlet.
fn unravel_qed_ns(res: Vec<Vec<Complex<f64>>>, order_qcd: usize, order_qed: usize) -> RawCmplx {
    let mut target = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };
    for col in res.iter().take(order_qcd + 1) {
        for el in col.iter().take(order_qed + 1) {
            target.re.push(el.re);
            target.im.push(el.im);
        }
    }
    target
}

/// Type for qcd gamma singlet return
type GammaQCDSinglet = PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>;

/// Type for qcd gamma ns return
type GammaQCDNS = PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>;

/// Type for OME return values.
type OmeResult = PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)>;

// TODO: once the Rust kernels grows we take be the computation of the integration path back to here
// in the old implementation it was ~
// // prepare Mellin stuff
// let path = mellin::TalbotPath::new(u, args.logx, is_singlet);
// let jac = path.jac() * path.prefactor();

/// Compute the QCD singlet anomalous dimension matrix in Mellin space.
///
/// # Errors
/// Returns `PyNotImplementedError` if `is_time_like` is set (both polarized
/// and unpolarized time-like are not yet implemented).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn qcd_gamma_singlet(
    py: Python<'_>,
    is_polarized: bool,
    is_time_like: bool,
    order_qcd: usize,
    re_n: f64,
    im_n: f64,
    nf: u8,
    variations: [u8; 4],
) -> GammaQCDSinglet {
    if is_polarized && is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized time-like is not implemented",
        ));
    }
    if is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Unpolarized time-like is not yet implemented",
        ));
    }
    let mut c = Cache::new(Complex::new(re_n, im_n));
    let raw = if is_polarized {
        unravel(
            ad_ps::gamma_singlet_qcd(order_qcd, &mut c, nf, variations),
            order_qcd,
        )
    } else {
        unravel(
            ad_us::gamma_singlet_qcd(order_qcd, &mut c, nf, variations),
            order_qcd,
        )
    };
    Ok((
        raw.re.into_pyarray_bound(py).unbind(),
        raw.im.into_pyarray_bound(py).unbind(),
    ))
}

/// Compute the QCD non-singlet anomalous dimension in Mellin space.
///
/// # Errors
/// Returns `PyNotImplementedError` for polarized time-like (not implemented)
/// or plain time-like (not yet implemented).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn qcd_gamma_ns(
    py: Python<'_>,
    is_polarized: bool,
    is_time_like: bool,
    order_qcd: usize,
    mode0: u16,
    re_n: f64,
    im_n: f64,
    nf: u8,
    variations: [u8; 3],
    _use_fhmruvv: bool, // TODO
) -> GammaQCDNS {
    if is_polarized && is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized time-like is not implemented",
        ));
    }
    if is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Unpolarized time-like is not yet implemented",
        ));
    }

    let mut c = Cache::new(Complex::new(re_n, im_n));

    let raw: Vec<Complex<f64>> = if is_polarized {
        ad_ps::gamma_ns_qcd(order_qcd, mode0, &mut c, nf, variations)
    } else {
        ad_us::gamma_ns_qcd(order_qcd, mode0, &mut c, nf, variations)
    };

    let re: Vec<f64> = raw.iter().map(|z| z.re).collect();
    let im: Vec<f64> = raw.iter().map(|z| z.im).collect();

    Ok((
        re.into_pyarray_bound(py).unbind(),
        im.into_pyarray_bound(py).unbind(),
    ))
}

/// Compute the singlet operator matrix element (OME) array in Mellin space.
///
/// # Errors
/// Returns a `PyNotImplementedError` in the following cases:
/// * Polarized time-like (not implemented).
/// * Polarized spacelike (not yet implemented).
/// * Unpolarized time-like (not yet implemented).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ome_singlet(
    py: Python<'_>,
    is_polarized: bool,
    is_time_like: bool,
    order_qcd: usize,
    re_n: f64,
    im_n: f64,
    nf: u8,
    l: f64,
    _is_msbar: bool,
) -> OmeResult {
    if is_polarized && is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized time-like OME singlet is not implemented",
        ));
    }
    if is_polarized {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized spacelike OME singlet is not yet implemented in Rust",
        ));
    }
    if is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Unpolarized time-like OME singlet is not yet implemented in Rust",
        ));
    }
    let mut c = Cache::new(Complex::new(re_n, im_n));
    let raw = unravel::<3>(ome_us::A_singlet(order_qcd, &mut c, nf, l), order_qcd);
    Ok((
        raw.re.into_pyarray_bound(py).unbind(),
        raw.im.into_pyarray_bound(py).unbind(),
    ))
}

/// Compute the non-singlet operator matrix element (OME) array **A** in Mellin space.
///
/// # Errors
/// Returns a `PyNotImplementedError` in the following cases:
/// * Polarized time-like (not implemented).
/// * Polarized spacelike (not yet implemented).
/// * Unpolarized time-like (not yet implemented).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ome_non_singlet(
    py: Python<'_>,
    is_polarized: bool,
    is_time_like: bool,
    matching_order_qcd: usize,
    re_n: f64,
    im_n: f64,
    nf: u8,
    l: f64,
) -> OmeResult {
    if is_polarized && is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized time-like OME non-singlet is not implemented",
        ));
    }
    if is_polarized {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polarized spacelike OME non-singlet is not yet implemented in Rust",
        ));
    }
    if is_time_like {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Unpolarized time-like OME non-singlet is not yet implemented in Rust",
        ));
    }
    let mut c = Cache::new(Complex::new(re_n, im_n));
    let raw = unravel::<2>(
        ome_us::A_non_singlet(matching_order_qcd, &mut c, nf, l),
        matching_order_qcd,
    );
    Ok((
        raw.re.into_pyarray_bound(py).unbind(),
        raw.im.into_pyarray_bound(py).unbind(),
    ))
}

/// Python extension module exposing EKO's Rust kernels.
///
/// Currently exposes:
/// - [`qcd_gamma_singlet`]: QCD singlet anomalous dimension matrix in Mellin space
/// - [`qcd_gamma_ns`]: QCD non-singlet anomalous dimension array in Mellin space
/// - [`ome_singlet`]: Singlet OME in Mellin space
/// - [`ome_non_singlet`]: Non Singlet OME in Mellin space
#[pymodule]
fn ekors(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qcd_gamma_singlet, m)?)?;
    m.add_function(wrap_pyfunction!(qcd_gamma_ns, m)?)?;
    m.add_function(wrap_pyfunction!(ome_singlet, m)?)?;
    m.add_function(wrap_pyfunction!(ome_non_singlet, m)?)?;
    Ok(())
}
