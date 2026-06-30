//! C-language interface for `ekore`.

mod ad_ps;
mod ad_us;
mod ome_us;

use ekore::harmonics::cache::Cache;
use num::complex::Complex;

/// C-compatible representation of a double-precision complex number.
///
/// The memory layout (`re` followed by `im`) matches `num::Complex<f64>` and
/// the C99 `double complex` / Fortran `COMPLEX(KIND=8)` types.
#[repr(C)]
pub struct ComplexF64 {
    pub re: f64,
    pub im: f64,
}

impl From<Complex<f64>> for ComplexF64 {
    fn from(c: Complex<f64>) -> Self {
        Self { re: c.re, im: c.im }
    }
}

/// Create a new `Cache` at Mellin N = `n_re + i·n_im`.
///
/// The returned pointer is heap-allocated and **must** be freed with [`cache_delete`].
#[no_mangle]
pub extern "C" fn cache_new(n_re: f64, n_im: f64) -> *mut Cache {
    Box::into_raw(Box::new(Cache::new(Complex::new(n_re, n_im))))
}

/// Free a `Cache` previously created with [`cache_new`].
///
/// Passing `NULL` is safe and does nothing.
///
/// # Safety
/// `c` must be a pointer returned by [`cache_new`] that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn cache_delete(c: *mut Cache) {
    if !c.is_null() {
        drop(Box::from_raw(c));
    }
}
