//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::{ComplexF64, MAX_ORDER_QCD, MAX_ORDER_QED};
use ekore::anomalous_dimensions::unpolarized::spacelike;
use ekore::harmonics::cache::Cache;

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
    result_len_body!(order_qcd > MAX_ORDER_QCD, order_qcd)
}

/// Compute the tower of the non-singlet anomalous dimensions.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of three `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `mode`: The specific non-singlet sector.
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    gamma_ns_qcd_body!(
        order_qcd,
        mode,
        c,
        nf,
        n3lo_variation,
        result,
        order_qcd > MAX_ORDER_QCD,
        spacelike::gamma_ns_qcd
    )
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
    result_len_body!(order_qcd > MAX_ORDER_QCD, order_qcd * 4)
}

/// Compute the tower of the singlet anomalous dimension matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of four `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_singlet_qcd(
    order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    gamma_singlet_qcd_body!(
        order_qcd,
        c,
        nf,
        n3lo_variation,
        result,
        order_qcd > MAX_ORDER_QCD,
        spacelike::gamma_singlet_qcd
    )
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
    result_len_body!(
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        (order_qcd + 1) * (order_qed + 1)
    )
}

/// Compute the tower of the |QCD| x |QED| non-singlet anomalous dimensions.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of three `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `mode`: The specific non-singlet sector.
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_ns_qed(
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    gamma_ns_qed_body!(
        order_qcd,
        order_qed,
        mode,
        c,
        nf,
        n3lo_variation,
        result,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_ns_qed
    )
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
    result_len_body!(
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        (order_qcd + 1) * (order_qed + 1) * 16
    )
}

/// Compute the tower of the |QCD| x |QED| singlet anomalous dimensions matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of seven `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_singlet_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    gamma_qed_matrix_body!(
        order_qcd,
        order_qed,
        c,
        nf,
        n3lo_variation,
        result,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_singlet_qed,
        4,
        16,
        7
    )
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
    result_len_body!(
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        (order_qcd + 1) * (order_qed + 1) * 4
    )
}

/// Compute the tower of the |QCD| x |QED| valence anomalous dimensions matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of three `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be `<= MAX_ORDER_QCD`).
/// * `order_qed`: The QED coupling power (must be `<= MAX_ORDER_QED`).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `result`: Pointer to the output buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_us_gamma_valence_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    gamma_qed_matrix_body!(
        order_qcd,
        order_qed,
        c,
        nf,
        n3lo_variation,
        result,
        order_qcd > MAX_ORDER_QCD || order_qed > MAX_ORDER_QED,
        spacelike::gamma_valence_qed,
        2,
        4,
        3
    )
}
