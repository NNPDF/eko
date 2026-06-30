//! The polarized, space-like anomalous dimensions.

use crate::ComplexF64;
use ekore::anomalous_dimensions::polarized::spacelike;
use ekore::harmonics::cache::Cache;
use std::slice;

/// Compute the tower of non-singlet anomalous dimensions.
#[no_mangle]
pub unsafe extern "C" fn ad_ps_gamma_ns_qcd(
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

/// Compute the tower of singlet anomalous dimension matrices.
#[no_mangle]
pub unsafe extern "C" fn ad_ps_gamma_singlet_qcd(
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
