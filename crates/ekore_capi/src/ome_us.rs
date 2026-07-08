//! The unpolarized, space-like |OME| at various couplings power.
#![allow(non_snake_case)]
use crate::ComplexF64;
use ekore::harmonics::cache::Cache;
use ekore::operator_matrix_elements::unpolarized::spacelike;
use std::slice;

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
    if matching_order_qcd >= 3 {
        return 0;
    }
    matching_order_qcd * 9
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
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ome_us_A_singlet_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ome_us_A_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || result.is_null() {
        return;
    }

    if matching_order_qcd >= 3 {
        return;
    }

    if result_len < matching_order_qcd * 9 {
        return;
    }

    unsafe {
        let c = &mut *c;
        let out = slice::from_raw_parts_mut(result, matching_order_qcd * 9);

        for (o, mat) in spacelike::A_singlet(matching_order_qcd, c, nf, L)
            .iter()
            .take(matching_order_qcd)
            .enumerate()
        {
            for r in 0..3_usize {
                for col in 0..3_usize {
                    out[o * 9 + r * 3 + col] = mat[r][col].into();
                }
            }
        }
    }
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
    if matching_order_qcd >= 3 {
        return 0;
    }
    matching_order_qcd * 4
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
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ome_us_A_non_singlet_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ome_us_A_non_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || result.is_null() {
        return;
    }

    if matching_order_qcd >= 3 {
        return;
    }

    if result_len < matching_order_qcd * 4 {
        return;
    }

    unsafe {
        let c = &mut *c;
        let out = slice::from_raw_parts_mut(result, matching_order_qcd * 4);

        for (o, mat) in spacelike::A_non_singlet(matching_order_qcd, c, nf, L)
            .iter()
            .take(matching_order_qcd)
            .enumerate()
        {
            for r in 0..2_usize {
                for col in 0..2_usize {
                    out[o * 4 + r * 2 + col] = mat[r][col].into();
                }
            }
        }
    }
}
