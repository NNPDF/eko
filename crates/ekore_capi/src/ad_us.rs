//! The unpolarized, space-like anomalous dimensions at various couplings power.

use crate::ComplexF64;
use ekore::anomalous_dimensions::unpolarized::spacelike;
use ekore::harmonics::cache::Cache;
use std::slice;

/// Compute the tower of the non-singlet anomalous dimensions.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
#[no_mangle]
pub unsafe extern "C" fn ad_us_gamma_ns_qcd(
    order_qcd: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
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

/// Compute the tower of the singlet anomalous dimension matrices.
///
/// `n3lo_variation = (gg, gq, qg, qq)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
#[no_mangle]
pub unsafe extern "C" fn ad_us_gamma_singlet_qcd(
    order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let var: [u8; 4] = slice::from_raw_parts(n3lo_variation, 4).try_into().unwrap();
    let out = slice::from_raw_parts_mut(result, order_qcd * 4);

    for (o, mat) in spacelike::gamma_singlet_qcd(order_qcd, c, nf, var)
        .iter()
        .enumerate()
    {
        for r in 0..2_usize {
            for col in 0..2_usize {
                out[o * 4 + r * 2 + col] = mat[r][col].into();
            }
        }
    }
}

/// Compute the tower of the |QCD| x |QED| non-singlet anomalous dimensions.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
#[no_mangle]
pub unsafe extern "C" fn ad_us_gamma_ns_qed(
    order_qcd: usize,
    order_qed: usize,
    mode: u16,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let var: [u8; 3] = slice::from_raw_parts(n3lo_variation, 3).try_into().unwrap();
    let ncols = order_qed + 1;
    let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * ncols);

    let gamma = spacelike::gamma_ns_qed(order_qcd, order_qed, mode, c, nf, var);
    for (i, row) in gamma.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            out[i * ncols + j] = (*val).into();
        }
    }
}

/// Compute the tower of the |QCD| x |QED| singlet anomalous dimensions matrices.
///
/// `n3lo_variation = (gg, gq, qg, qq, ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
#[no_mangle]
pub unsafe extern "C" fn ad_us_gamma_singlet_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let var: [u8; 7] = slice::from_raw_parts(n3lo_variation, 7).try_into().unwrap();
    let ncols = order_qed + 1;
    let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * ncols * 16);

    let gamma = spacelike::gamma_singlet_qed(order_qcd, order_qed, c, nf, var);
    for (i, row) in gamma.iter().enumerate() {
        for (j, mat) in row.iter().enumerate() {
            let base = (i * ncols + j) * 16;
            for r in 0..4_usize {
                for col in 0..4_usize {
                    out[base + r * 4 + col] = mat[r][col].into();
                }
            }
        }
    }
}

/// Compute the tower of the |QCD| x |QED| valence anomalous dimensions matrices.
///
/// `n3lo_variation = (ns_p, ns_m, ns_v)` is a list indicating which variation should
/// be used. `variation = 1,2` is the upper/lower bound, while any other value
/// returns the central (averaged) value.
#[no_mangle]
pub unsafe extern "C" fn ad_us_gamma_valence_qed(
    order_qcd: usize,
    order_qed: usize,
    c: *mut Cache,
    nf: u8,
    n3lo_variation: *const u8,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let var: [u8; 3] = slice::from_raw_parts(n3lo_variation, 3).try_into().unwrap();
    let ncols = order_qed + 1;
    let out = slice::from_raw_parts_mut(result, (order_qcd + 1) * ncols * 4);

    let gamma = spacelike::gamma_valence_qed(order_qcd, order_qed, c, nf, var);
    for (i, row) in gamma.iter().enumerate() {
        for (j, mat) in row.iter().enumerate() {
            let base = (i * ncols + j) * 4;
            for r in 0..2_usize {
                for col in 0..2_usize {
                    out[base + r * 2 + col] = mat[r][col].into();
                }
            }
        }
    }
}
