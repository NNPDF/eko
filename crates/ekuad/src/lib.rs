use std::ffi::c_void;

use ekore;

#[no_mangle]
pub unsafe extern "C" fn quad_ker(x: f64, extra: *mut c_void) -> f64 {
    let ex = *(extra as *mut Extra);
    ekore::ciao(x * ex.slope, ex.shift)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Extra {
    pub slope: f64,
    pub shift: f64,
}

/// This is required to make `Extra` part of the API, otherwise it won't be added to the compiled
/// package (since it does not appear in the signature of `quad_ker`).
#[no_mangle]
pub unsafe extern "C" fn dummy() -> Extra {
    Extra {
        slope: 0.,
        shift: 0.,
    }
}
