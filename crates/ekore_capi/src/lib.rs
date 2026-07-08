//! C-language interface for `ekore`.

pub mod ad_ps;
pub mod ad_us;
pub mod ome_us;

use ekore::harmonics::cache::Cache as EkoreCache;
use num::complex::Complex;

/// Maximum QCD coupling power implemented.
pub const MAX_ORDER_QCD: usize = 4;
/// Maximum QED coupling power implemented.
pub const MAX_ORDER_QED: usize = 2;

/// singlet-like non-singlet |PID|.
pub const PID_NSP: u16 = 10101;
/// valence-like non-singlet |PID|.
pub const PID_NSM: u16 = 10201;
/// non-singlet all-valence |PID|.
pub const PID_NSV: u16 = 10200;
/// singlet-like non-singlet up-sector |PID|.
pub const PID_NSP_U: u16 = 10102;
/// singlet-like non-singlet down-sector |PID|.
pub const PID_NSP_D: u16 = 10103;
/// valence-like non-singlet up-sector |PID|.
pub const PID_NSM_U: u16 = 10202;
/// valence-like non-singlet down-sector |PID|.
pub const PID_NSM_D: u16 = 10203;

// Compile-time assertions to ensure C API constants remain synced with `ekore`
const _: () = {
    assert!(MAX_ORDER_QCD == ekore::constants::MAX_ORDER_QCD);
    assert!(MAX_ORDER_QED == ekore::constants::MAX_ORDER_QED);
    assert!(PID_NSP == ekore::constants::PID_NSP);
    assert!(PID_NSM == ekore::constants::PID_NSM);
    assert!(PID_NSV == ekore::constants::PID_NSV);
    assert!(PID_NSP_U == ekore::constants::PID_NSP_U);
    assert!(PID_NSP_D == ekore::constants::PID_NSP_D);
    assert!(PID_NSM_U == ekore::constants::PID_NSM_U);
    assert!(PID_NSM_D == ekore::constants::PID_NSM_D);
};

/// C-compatible representation of a double-precision complex number.
///
/// The memory layout (`re` followed by `im`) matches `num::Complex<f64>` type.
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

/// Opaque handle to the Mellin-space harmonics cache.
/// Create with `cache_new`, free with `cache_delete`.
pub struct Cache;

/// Create a new `Cache` at Mellin N = `n_re` + i·`n_im`.
///
/// The returned pointer is heap-allocated and **must** be freed with [`cache_delete`].
///
/// # Parameters
/// * n_re: The real part of the Mellin variable N.
/// * n_im: The imaginary part of the Mellin variable N.
///
/// # Returns
/// * Returns a raw pointer to a newly allocated `Cache`.
#[unsafe(no_mangle)]
pub extern "C" fn cache_new(n_re: f64, n_im: f64) -> *mut Cache {
    let real_cache = EkoreCache::new(Complex::new(n_re, n_im));
    Box::into_raw(Box::new(real_cache)) as *mut Cache
}

/// Free a `Cache` previously created with [`cache_new`].
///
/// Passing `NULL` is safe and does nothing.
///
/// # Safety
/// * `c` must be a valid pointer returned by [`cache_new`] that has not already been freed.
///   Double-freeing or passing arbitrary pointers will result in undefined behavior.
///
/// # Parameters
/// * c: Pointer to the `Cache` to be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cache_delete(c: *mut Cache) {
    if !c.is_null() {
        unsafe {
            drop(Box::from_raw(c as *mut EkoreCache));
        }
    }
}
