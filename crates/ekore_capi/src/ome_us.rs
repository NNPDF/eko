//! The unpolarized, space-like |OME| at various couplings power.
#![allow(non_snake_case)]
use crate::ComplexF64;
use ekore::harmonics::cache::Cache;
use ekore::operator_matrix_elements::unpolarized::spacelike;

/// Required length of `result` for [`ome_us_A_singlet`] at the given `matching_order_qcd`.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (must be < 3).
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `matching_order_qcd` is out of the supported range.
#[unsafe(no_mangle)]
pub extern "C" fn ome_us_A_singlet_result_len(matching_order_qcd: usize) -> usize {
    result_len_body!(matching_order_qcd >= 3, matching_order_qcd * 9)
}

/// Compute the tower of the singlet |OME|.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
/// * The `result` buffer must have a capacity of at least `matching_order_qcd * 9` elements.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (supported range: < 3).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `L`: The logarithm parameter.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ome_us_A_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
) {
    ome_matrix_body!(
        matching_order_qcd,
        c,
        nf,
        L,
        result,
        matching_order_qcd >= 3,
        spacelike::A_singlet,
        3,
        9
    )
}

/// Required length of `result` for [`ome_us_A_non_singlet`] at the given `matching_order_qcd`.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (must be < 3).
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `matching_order_qcd` is out of the supported range.
#[unsafe(no_mangle)]
pub extern "C" fn ome_us_A_non_singlet_result_len(matching_order_qcd: usize) -> usize {
    result_len_body!(matching_order_qcd >= 3, matching_order_qcd * 4)
}

/// Compute the tower of the non-singlet |OME|.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
/// * The `result` buffer must have a capacity of at least `matching_order_qcd * 4` elements.
///
/// # Parameters
/// * `matching_order_qcd`: The QCD matching order (supported range: < 3).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `L`: The logarithm parameter.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ome_us_A_non_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
) {
    ome_matrix_body!(
        matching_order_qcd,
        c,
        nf,
        L,
        result,
        matching_order_qcd >= 3,
        spacelike::A_non_singlet,
        2,
        4
    )
}
