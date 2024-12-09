//! Interface to the eko Python package.

use ekore::harmonics::cache::Cache;
use num::Complex;
use std::ffi::c_void;

pub mod bib;
pub mod mellin;

/// Wrapper to pass arguments back to Python
struct RawCmplx {
    re: Vec<f64>,
    im: Vec<f64>,
}

/// Map tensors to c-ordered list
/// (res is a vector with dim order_qcd filled with DIMxDIM matrices)
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

/// Map tensors to c-ordered list in the QED singlet and valence case
/// (res is a matrix with dim order_qcd x order_qed filled with DIMxDIM matrices)
fn unravel_qed<const DIM: usize>(
    res: Vec<Vec<[[Complex<f64>; DIM]; DIM]>>,
    order_qcd: usize,
    order_qed: usize,
) -> RawCmplx {
    let mut target = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };
    for obj_ in res.iter().take(order_qcd) {
        for obj in obj_.iter().take(order_qed) {
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

/// Map tensors to c-ordered list in the QED non-singlet case
/// (res is a matrix with dim order_qcd x order_qed filled with complex numbers)
fn unravel_qed_ns(res: Vec<Vec<Complex<f64>>>, order_qcd: usize, order_qed: usize) -> RawCmplx {
    let mut target = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };
    for col in res.iter().take(order_qcd) {
        for el in col.iter().take(order_qed) {
            target.re.push(el.re);
            target.im.push(el.im);
        }
    }
    target
}

/// QCD intergration kernel inside quad.
///
/// # Safety
/// This is the connection from Python, so we don't know what is on the other side.
#[no_mangle]
pub unsafe extern "C" fn rust_quad_ker_qcd(u: f64, rargs: *mut c_void) -> f64 {
    let args = *(rargs as *mut QuadQCDargs);

    let is_singlet = (100 == args.mode0)
        || (21 == args.mode0)
        || (90 == args.mode0)
        || (22 == args.mode0)
        || (101 == args.mode0);
    let is_qed_valence = (10200 == args.mode0) || (10204 == args.mode0);
    // prepare Mellin stuff
    let path = mellin::TalbotPath::new(u, args.logx, is_singlet);
    let jac = path.jac() * path.prefactor();
    let mut c = Cache::new(path.n());
    let mut raw = RawCmplx {
        re: Vec::<f64>::new(),
        im: Vec::<f64>::new(),
    };

    if args.is_ome {
        if is_singlet {
            raw = unravel(
                ekore::operator_matrix_elements::unpolarized::spacelike::A_singlet(
                    args.order_qcd,
                    &mut c,
                    args.nf,
                    args.L,
                ),
                args.order_qcd,
            );
        } else {
            raw = unravel(
                ekore::operator_matrix_elements::unpolarized::spacelike::A_non_singlet(
                    args.order_qcd,
                    &mut c,
                    args.nf,
                    args.L,
                ),
                args.order_qcd,
            );
        }
    } else if is_singlet {
        if args.order_qed > 0 {
            let gamma_singlet_qed =
                ekore::anomalous_dimensions::unpolarized::spacelike::gamma_singlet_qed;
            raw = unravel_qed(
                gamma_singlet_qed(args.order_qcd, args.order_qed, &mut c, args.nf),
                args.order_qcd,
                args.order_qed,
            );
        } else {
            let gamma_singlet_qcd = match args.is_polarized {
                true => ekore::anomalous_dimensions::polarized::spacelike::gamma_singlet_qcd,
                false => ekore::anomalous_dimensions::unpolarized::spacelike::gamma_singlet_qcd,
            };
            raw = unravel(
                gamma_singlet_qcd(args.order_qcd, &mut c, args.nf),
                args.order_qcd,
            );
        }
    } else if args.order_qed > 0 {
        if is_qed_valence {
            let gamma_valence_qed =
                ekore::anomalous_dimensions::unpolarized::spacelike::gamma_valence_qed;
            raw = unravel_qed(
                gamma_valence_qed(args.order_qcd, args.order_qed, &mut c, args.nf),
                args.order_qcd,
                args.order_qed,
            );
        } else {
            let gamma_ns_qed = ekore::anomalous_dimensions::unpolarized::spacelike::gamma_ns_qed;
            raw = unravel_qed_ns(
                gamma_ns_qed(args.order_qcd, args.order_qed, args.mode0, &mut c, args.nf),
                args.order_qcd,
                args.order_qed,
            );
        }
    } else {
        // we can not do 1D
        let gamma_ns_qcd = match args.is_polarized {
            true => ekore::anomalous_dimensions::polarized::spacelike::gamma_ns_qcd,
            false => ekore::anomalous_dimensions::unpolarized::spacelike::gamma_ns_qcd,
        };
        let res = gamma_ns_qcd(args.order_qcd, args.mode0, &mut c, args.nf);
        for el in res.iter().take(args.order_qcd) {
            raw.re.push(el.re);
            raw.im.push(el.im);
        }
    }

    // pass on
    (args.py)(
        raw.re.as_ptr(),
        raw.im.as_ptr(),
        c.n().re,
        c.n().im,
        jac.re,
        jac.im,
        args.order_qcd,
        args.order_qed,
        is_singlet,
        args.mode0,
        args.mode1,
        args.nf,
        args.is_log,
        args.logx,
        args.areas,
        args.areas_x,
        args.areas_y,
        args.L,
        args.method_num,
        args.as1,
        args.as0,
        args.ev_op_iterations,
        args.ev_op_max_order_qcd,
        args.sv_mode_num,
        args.is_threshold,
        args.Lsv,
        // additional QED params
        args.as_list,
        args.as_list_len,
        args.mu2_from,
        args.mu2_to,
        args.a_half,
        args.a_half_x,
        args.a_half_y,
        args.alphaem_running,
    )
}

/// Python callback signature
type PyQuadKerQCDT = unsafe extern "C" fn(
    *const f64,
    *const f64,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    bool,
    u16,
    u16,
    u8,
    bool,
    f64,
    *const f64,
    u8,
    u8,
    f64,
    u8,
    f64,
    f64,
    u8,
    u8,
    u8,
    bool,
    f64,
    *const f64,
    u8,
    f64,
    f64,
    *const f64,
    u8,
    u8,
    bool,
) -> f64;

/// Additional integration parameters
#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuadQCDargs {
    pub order_qcd: usize,
    pub order_qed: usize,
    pub mode0: u16,
    pub mode1: u16,
    pub is_polarized: bool,
    pub is_time_like: bool,
    pub nf: u8,
    pub py: PyQuadKerQCDT,
    pub is_log: bool,
    pub logx: f64,
    pub areas: *const f64,
    pub areas_x: u8,
    pub areas_y: u8,
    pub L: f64,
    pub method_num: u8,
    pub as1: f64,
    pub as0: f64,
    pub ev_op_iterations: u8,
    pub ev_op_max_order_qcd: u8,
    pub sv_mode_num: u8,
    pub is_threshold: bool,
    pub is_ome: bool,
    pub Lsv: f64,
    // additional param required for QED
    pub as_list: *const f64,
    pub as_list_len: u8,
    pub mu2_from: f64,
    pub mu2_to: f64,
    pub a_half: *const f64,
    pub a_half_x: u8,
    pub a_half_y: u8,
    pub alphaem_running: bool,
}

/// Empty placeholder function for python callback.
///
/// # Safety
/// This is the connection back to Python, so we don't know what is on the other side.
pub unsafe extern "C" fn my_py(
    _re_gamma: *const f64,
    _im_gamma: *const f64,
    _re_n: f64,
    _im_n: f64,
    _re_jac: f64,
    _im_jac: f64,
    _order_qcd: usize,
    _order_qed: usize,
    _is_singlet: bool,
    _mode0: u16,
    _mode1: u16,
    _nf: u8,
    _is_log: bool,
    _logx: f64,
    _areas: *const f64,
    _areas_x: u8,
    _areas_y: u8,
    _l: f64,
    _method_num: u8,
    _as1: f64,
    _as0: f64,
    _ev_op_iterations: u8,
    _ev_op_max_order_qcd: u8,
    _sv_mode_num: u8,
    _is_threshold: bool,
    _lsv: f64,
    _as_list: *const f64,
    _as_list_len: u8,
    _mu2_from: f64,
    _mu2_to: f64,
    _a_half: *const f64,
    _a_half_x: u8,
    _a_half_y: u8,
    _alphaem_running: bool,
) -> f64 {
    0.
}

/// Return empty additional arguments.
///
/// This is required to make the arguments part of the API, otherwise it won't be added to the compiled
/// package (since it does not appear in the signature of `rust_quad_ker_qcd`).
///
/// # Safety
/// This is the connection from and back to Python, so we don't know what is on the other side.
#[no_mangle]
pub unsafe extern "C" fn empty_qcd_args() -> QuadQCDargs {
    QuadQCDargs {
        order_qcd: 0,
        order_qed: 0,
        mode0: 0,
        mode1: 0,
        is_polarized: false,
        is_time_like: false,
        nf: 0,
        py: my_py,
        is_log: true,
        logx: 0.,
        areas: [].as_ptr(),
        areas_x: 0,
        areas_y: 0,
        L: 0.,
        method_num: 0,
        as1: 0.,
        as0: 0.,
        ev_op_iterations: 0,
        ev_op_max_order_qcd: 0,
        sv_mode_num: 0,
        is_threshold: false,
        is_ome: false,
        Lsv: 0.,
        as_list: [].as_ptr(),
        as_list_len: 0,
        mu2_from: 0.,
        mu2_to: 0.,
        a_half: [].as_ptr(),
        a_half_x: 0,
        a_half_y: 0,
        alphaem_running: false,
    }
}
