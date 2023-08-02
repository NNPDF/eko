use std::ffi::c_void;

use ekore;

#[no_mangle]
pub unsafe extern "C" fn quad_ker(x: f64, extra: *mut c_void) -> f64 {
    let ex = *(extra as *mut Extra);
    ekore::ciao(x * ex.slope, ex.shift);
    let ar = [0.1,0.023];
    (ex.py)(x * ex.slope, ex.shift, ar.as_ptr(), 2, ex.areas, ex.areas_len)
}

type PyT = unsafe extern "C" fn(f64, f64, *const f64, u16, *const f64, u16)-> f64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Extra {
    pub slope: f64,
    pub shift: f64,
    pub py: PyT,
    pub areas: *const f64,
    pub areas_len: u16
}

pub unsafe extern "C" fn my_py(_x: f64, _y: f64, _ar: *const f64, _len: u16, _areas: *const f64, _areas_len: u16) -> f64 {
    0.
}

/// This is required to make `Extra` part of the API, otherwise it won't be added to the compiled
/// package (since it does not appear in the signature of `quad_ker`).
#[no_mangle]
pub unsafe extern "C" fn dummy() -> Extra {
    Extra {
        slope: 0.,
        shift: 0.,
        py: my_py,
        areas: [].as_ptr(),
        areas_len: 0
    }
}
