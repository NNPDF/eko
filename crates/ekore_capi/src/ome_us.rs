//! The unpolarized, space-like |OME| at various couplings power.
#![allow(non_snake_case)]
use crate::ComplexF64;
use ekore::harmonics::cache::Cache;
use ekore::operator_matrix_elements::unpolarized::spacelike;
use std::slice;

/// Compute the tower of the singlet |OME|.
#[no_mangle]
pub unsafe extern "C" fn ome_us_A_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let out = slice::from_raw_parts_mut(result, matching_order_qcd * 9);

    for (o, mat) in spacelike::A_singlet(matching_order_qcd, c, nf, L)
        .iter()
        .enumerate()
    {
        for r in 0..3_usize {
            for col in 0..3_usize {
                out[o * 9 + r * 3 + col] = mat[r][col].into();
            }
        }
    }
}

/// Compute the tower of the non-singlet |OME|.
#[no_mangle]
pub unsafe extern "C" fn ome_us_A_non_singlet(
    matching_order_qcd: usize,
    c: *mut Cache,
    nf: u8,
    L: f64,
    result: *mut ComplexF64,
) {
    let c = &mut *c;
    let out = slice::from_raw_parts_mut(result, matching_order_qcd * 4);

    for (o, mat) in spacelike::A_non_singlet(matching_order_qcd, c, nf, L)
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
