//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::{
    ComplexF64, MAX_ORDER_QCD, MAX_ORDER_QED, PID_NSM, PID_NSM_D, PID_NSM_U, PID_NSP, PID_NSP_D,
    PID_NSP_U, PID_NSV,
};
use ekore::anomalous_dimensions::unpolarized::spacelike;
use ekore::harmonics::cache::Cache;
use std::slice;

/// Required length of `n3lo_variation` for [`ad_us_gamma_ns_qcd`].
///
/// # Returns
/// * Returns the fixed required buffer length of `3`.
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_ns_qcd_n3lo_len() -> usize {
    3
}

/// Required length of `result` for [`ad_us_gamma_ns_qcd`] at the given `order_qcd`.
///
/// # Parameters
/// * `order_qcd`: The QCD matching order.
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `order_qcd` is out of the supported range (i.e., `> MAX_ORDER_QCD`).
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_ns_qcd_result_len(order_qcd: usize) -> usize {
    if order_qcd > MAX_ORDER_QCD {
        return 0;
    }
    order_qcd
}

/// Compute the tower of the non-singlet anomalous dimensions.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `mode`: The specific non-singlet sector.
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_us_gamma_ns_qcd_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should be at least the value returned by [`ad_us_gamma_ns_qcd_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    n3lo_len: usize,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || n3lo_variation.is_null() || result.is_null() {
        return;
    }

    if order_qcd > MAX_ORDER_QCD {
        return;
    }

    if !matches!(mode, PID_NSP | PID_NSM | PID_NSV) {
        return;
    }

    if n3lo_len < 3 || result_len < order_qcd {
        return;
    }

    unsafe {
        let c = &mut *c;
        let var: [u8; 3] = slice::from_raw_parts(n3lo_variation, 3).try_into().unwrap();
        let out = slice::from_raw_parts_mut(result, order_qcd);

        for (dst, src) in out
            .iter_mut()
            .zip(spacelike::gamma_ns_qcd(order_qcd, mode, c, nf, var))
        {
            *dst = src.into();
        }
    }
}

/// Required length of `n3lo_variation` for [`ad_us_gamma_singlet_qcd`].
///
/// # Returns
/// * Returns the fixed required buffer length of `4`.
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_singlet_qcd_n3lo_len() -> usize {
    4
}

/// Required length of `result` for [`ad_us_gamma_singlet_qcd`] at the given `order_qcd`.
///
/// # Parameters
/// * `order_qcd`: The QCD matching order.
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `order_qcd` is out of the supported range (i.e., `> MAX_ORDER_QCD`).
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_singlet_qcd_result_len(order_qcd: usize) -> usize {
    if order_qcd > MAX_ORDER_QCD {
        return 0;
    }
    order_qcd * 4
}

/// Compute the tower of the singlet anomalous dimension matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_us_gamma_singlet_qcd_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_us_gamma_singlet_qcd_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_singlet_qcd(
    order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    n3lo_len: usize,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || n3lo_variation.is_null() || result.is_null() {
        return;
    }

    if order_qcd > MAX_ORDER_QCD {
        return;
    }

    if n3lo_len < 4 || result_len < (order_qcd * 4) {
        return;
    }

    unsafe {
        let c = &mut *c;
        let var: [u8; 4] = slice::from_raw_parts(n3lo_variation, 4).try_into().unwrap();
        let out = slice::from_raw_parts_mut(result, order_qcd * 4);

        for (o, mat) in spacelike::gamma_singlet_qcd(order_qcd, c, nf, var)
            .iter()
            .take(order_qcd)
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

/// Required length of `n3lo_variation` for [`ad_us_gamma_ns_qed`].
///
/// # Returns
/// * Returns the fixed required buffer length of `3`.
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_ns_qed_n3lo_len() -> usize {
    3
}

/// Required length of `result` for [`ad_us_gamma_ns_qed`] at the given `order_qcd` and `order_qed`.
///
/// # Parameters
/// * `order_qcd`: The QCD matching order.
/// * `order_qed`: The QED matching order.
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if either order is out of the supported range (`> MAX_ORDER_QCD` or `> MAX_ORDER_QED`).
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_ns_qed_result_len(order_qcd: usize, order_qed: usize) -> usize {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return 0;
    }
    (order_qcd + 1) * (order_qed + 1)
}

/// Compute the tower of the |QCD| x |QED| non-singlet anomalous dimensions.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `mode`: The specific non-singlet sector.
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_us_gamma_ns_qed_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_us_gamma_ns_qed_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_ns_qed(
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    n3lo_len: usize,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || n3lo_variation.is_null() || result.is_null() {
        return;
    }

    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return;
    }

    if !matches!(
        mode,
        PID_NSP_U | PID_NSP_D | PID_NSM_U | PID_NSM_D | PID_NSP | PID_NSM | PID_NSV
    ) {
        return;
    }

    let required_result_len = (order_qcd + 1) * (order_qed + 1);

    if n3lo_len < 3 || result_len < required_result_len {
        return;
    }

    unsafe {
        let c = &mut *c;
        let var: [u8; 3] = slice::from_raw_parts(n3lo_variation, 3).try_into().unwrap();
        let ncols = order_qed + 1;
        let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * ncols);

        let gamma = spacelike::gamma_ns_qed(order_qcd, order_qed, mode, c, nf, var);
        for (i, row) in gamma.iter().take(order_qcd + 1).enumerate() {
            for (j, val) in row.iter().take(ncols).enumerate() {
                out[i * ncols + j] = (*val).into();
            }
        }
    }
}
/// Required length of `n3lo_variation` for [`ad_us_gamma_singlet_qed`].
///
/// # Returns
/// * Returns the fixed required buffer length of `7`.
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_singlet_qed_n3lo_len() -> usize {
    7
}

/// Required length of `result` for [`ad_us_gamma_singlet_qed`] at the given `order_qcd` and `order_qed`.
///
/// # Parameters
/// * `order_qcd`: The QCD matching order.
/// * `order_qed`: The QED matching order.
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if either order is out of the supported range (`> MAX_ORDER_QCD` or `> MAX_ORDER_QED`).
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_singlet_qed_result_len(order_qcd: usize, order_qed: usize) -> usize {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return 0;
    }
    (order_qcd + 1) * (order_qed + 1) * 16
}

/// Compute the tower of the |QCD| x |QED| singlet anomalous dimensions matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_us_gamma_singlet_qed_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_us_gamma_singlet_qed_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_singlet_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    n3lo_len: usize,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || n3lo_variation.is_null() || result.is_null() {
        return;
    }

    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return;
    }

    let required_result_len = (order_qcd + 1) * (order_qed + 1) * 16;

    if n3lo_len < 7 || result_len < required_result_len {
        return;
    }

    unsafe {
        let c = &mut *c;
        let var: [u8; 7] = slice::from_raw_parts(n3lo_variation, 7).try_into().unwrap();
        let ncols = order_qed + 1;
        let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * (order_qed + 1) * 16);

        let gamma = spacelike::gamma_singlet_qed(order_qcd, order_qed, c, nf, var);
        for (i, row) in gamma.iter().take(order_qcd + 1).enumerate() {
            for (j, mat) in row.iter().take(ncols).enumerate() {
                let base = (i * ncols + j) * 16;
                for r in 0..4_usize {
                    for col in 0..4_usize {
                        out[base + r * 4 + col] = mat[r][col].into();
                    }
                }
            }
        }
    }
}

/// Required length of `n3lo_variation` for [`ad_us_gamma_valence_qed`].
///
/// # Returns
/// * Returns the fixed required buffer length of `3`.
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_valence_qed_n3lo_len() -> usize {
    3
}

/// Required length of `result` for [`ad_us_gamma_valence_qed`] at the given `order_qcd` and `order_qed`.
///
/// # Parameters
/// * `order_qcd`: The QCD matching order.
/// * `order_qed`: The QED matching order.
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if either order is out of the supported range (`> MAX_ORDER_QCD` or `> MAX_ORDER_QED`).
#[unsafe(no_mangle)]
pub extern "C" fn ad_us_gamma_valence_qed_result_len(order_qcd: usize, order_qed: usize) -> usize {
    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return 0;
    }
    (order_qcd + 1) * (order_qed + 1) * 4
}

/// Compute the tower of the |QCD| x |QED| valence anomalous dimensions matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_us_gamma_valence_qed_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_us_gamma_valence_qed_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_valence_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    n3lo_len: usize,
    result: *mut ComplexF64,
    result_len: usize,
) {
    if c.is_null() || n3lo_variation.is_null() || result.is_null() {
        return;
    }

    if order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED {
        return;
    }

    let required_result_len = (order_qcd + 1) * (order_qed + 1) * 4;

    if n3lo_len < 3 || result_len < required_result_len {
        return;
    }

    unsafe {
        let c = &mut *c;
        let var: [u8; 3] = slice::from_raw_parts(n3lo_variation, 3).try_into().unwrap();
        let ncols = order_qed + 1;
        let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * ncols * 4);

        let gamma = spacelike::gamma_valence_qed(order_qcd, order_qed, c, nf, var);
        for (i, row) in gamma.iter().take(order_qcd + 1).enumerate() {
            for (j, mat) in row.iter().take(ncols).enumerate() {
                let base = (i * ncols + j) * 4;
                for r in 0..2_usize {
                    for col in 0..2_usize {
                        out[base + r * 2 + col] = mat[r][col].into();
                    }
                }
            }
        }
    }
}
