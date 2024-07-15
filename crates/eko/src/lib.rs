//! Interface to the eko Python package.

use ekore::harmonics::cache::Cache;
use std::ffi::c_void;

pub mod bib;
pub mod mellin;

/// QCD intergration kernel inside quad.
///
/// # Safety
/// This is the connection from Python, so we don't know what is on the other side.
#[no_mangle]
pub unsafe extern "C" fn rust_quad_ker_qcd(u: f64, rargs: *mut c_void) -> f64 {
    let args = *(rargs as *mut QuadQCDargs);
    let is_singlet = (100 == args.mode0) || (21 == args.mode0) || (90 == args.mode0);
    // prepare gamma
    let path = mellin::TalbotPath::new(u, args.logx, is_singlet);
    let jac = path.jac() * path.prefactor();
    let mut c = Cache::new(path.n());
    let mut re = Vec::<f64>::new();
    let mut im = Vec::<f64>::new();
    if is_singlet {
        let res = ekore::anomalous_dimensions::unpolarized::spacelike::gamma_singlet_qcd(
            args.order_qcd,
            &mut c,
            args.nf,
        );
        for gamma_s in res.iter().take(args.order_qcd) {
            for col in gamma_s.iter().take(2) {
                for el in col.iter().take(2) {
                    re.push(el.re);
                    im.push(el.im);
                }
            }
        }
    } else {
        let res = ekore::anomalous_dimensions::unpolarized::spacelike::gamma_ns_qcd(
            args.order_qcd,
            args.mode0,
            &mut c,
            args.nf,
        );
        for el in res.iter().take(args.order_qcd) {
            re.push(el.re);
            im.push(el.im);
        }
    }
    // pass on
    (args.py)(
        re.as_ptr(),
        im.as_ptr(),
        c.n.re,
        c.n.im,
        jac.re,
        jac.im,
        args.order_qcd,
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
) -> f64;

/// Additional integration parameters
#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuadQCDargs {
    pub order_qcd: usize,
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
) -> f64 {
    0.
}

/// Return empty additional arguments.
///
/// This is required to make the arguments part of the API, otherwise it won't be added to the compiled
/// package (since it does not appear in the signature of `quad_ker_qcd`).
///
/// # Safety
/// This is the connection from and back to Python, so we don't know what is on the other side.
#[no_mangle]
pub unsafe extern "C" fn empty_qcd_args() -> QuadQCDargs {
    QuadQCDargs {
        order_qcd: 0,
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
    }
}
