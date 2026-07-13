//! The polarized, space-like anomalous dimensions.

use crate::ComplexF64;
use ekore::anomalous_dimensions::polarized::spacelike;
use ekore::harmonics::cache::Cache;

/// Required length of `result` for [`ad_ps_gamma_ns_qcd`] at the given `order_qcd`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be < 3).
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `order_qcd` is out of the supported range.
#[unsafe(no_mangle)]
pub extern "C" fn ad_ps_gamma_ns_qcd_result_len(order_qcd: usize) -> usize {
    result_len_body!(order_qcd >= 3, order_qcd)
}

/// Compute the tower of non-singlet anomalous dimensions.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of three `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (supported range: < 3).
/// * `mode`: The specific non-singlet sector.
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_ps_gamma_ns_qcd_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_ps_gamma_ns_qcd_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_ps_gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
    result_len: usize,
) {
    gamma_ns_qcd_body!(
        order_qcd,
        mode,
        c,
        nf,
        n3lo_variation,
        result,
        result_len,
        order_qcd >= 3,
        spacelike::gamma_ns_qcd
    )
}

/// Required length of `result` for [`ad_ps_gamma_singlet_qcd`] at the given `order_qcd`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (must be < 3).
///
/// # Returns
/// * Returns the required buffer size (number of `ComplexF64` elements).
/// * Returns `0` if `order_qcd` is out of the supported range.
#[unsafe(no_mangle)]
pub extern "C" fn ad_ps_gamma_singlet_qcd_result_len(order_qcd: usize) -> usize {
    result_len_body!(order_qcd >= 3, order_qcd * 4)
}

/// Compute the tower of singlet anomalous dimension matrices.
///
/// # Safety
/// * `c` must be a valid, non-null pointer to an initialized `Cache`.
/// * `n3lo_variation` must be a valid, non-null pointer to a buffer of four `u8` elements.
/// * `result` must be a valid, non-null pointer to a contiguous, properly aligned buffer of `ComplexF64`.
///
/// # Parameters
/// * `order_qcd`: The QCD coupling power (supported range: < 3).
/// * `c`: Pointer to the harmonic cache.
/// * `nf`: Number of active flavors.
/// * `n3lo_variation`: Pointer to the buffer containing N3LO variations.
/// * `n3lo_len`: The actual length of the provided `n3lo_variation` buffer. This should be at
///   least the value returned by [`ad_ps_gamma_singlet_qcd_n3lo_len`].
/// * `result`: Pointer to the output buffer.
/// * `result_len`: The actual length (in elements) of the provided `result` buffer. This should
///   be at least the value returned by [`ad_ps_gamma_singlet_qcd_result_len`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ad_ps_gamma_singlet_qcd(
    order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
    result_len: usize,
) {
    gamma_singlet_qcd_body!(
        order_qcd,
        c,
        nf,
        n3lo_variation,
        result,
        result_len,
        order_qcd >= 3,
        spacelike::gamma_singlet_qcd
    )
}
