//! C-language interface for [`ekore`], the crate providing the anomalous dimensions and
//! operator matrix elements of the [EKO](https://github.com/NNPDF/eko) framework.
//!
//! This crate re-exposes those quantities through a `#[no_mangle]` C ABI, so they can be
//! called from C, C++, or any other language with a C FFI. See the main
//! [EKO documentation](https://eko.readthedocs.io/en/latest/) for the physics behind the
//! computed quantities, and [`ekore`] for the underlying Rust API.
//!
//! # Installation of pre-built library
//!
//! Instead of building from source, you can install a pre-built version of the library
//! (for Linux and macOS, on x86_64 and aarch64) by running
//!
//! ```sh
//! curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/NNPDF/eko/master/crates/ekore_capi/install-capi.sh | sh
//! ```
//!
//! You'll be prompted for an installation prefix (or pass one non-interactively with
//! `--prefix`). This installs the shared/static library, the C header, and a `pkg-config`
//! file, already patched with the chosen prefix. See [Consuming](#consuming) below for
//! how to use it.
//!
//! # Building from source
//!
//! Alternatively, you can build the crate from source:
//!
//! 1. Install [`cargo-c`](https://crates.io/crates/cargo-c), which is required to generate
//!    the C header and `pkg-config` file alongside the library:
//!
//!    ```sh
//!    cargo install cargo-c
//!    ```
//!
//! 2. Check out the [EKO repository](https://github.com/NNPDF/eko), then from its root run
//!
//!    ```sh
//!    cargo cinstall --release -p ekore_capi --prefix=${prefix} --libdir=${prefix}/lib
//!    ```
//!
//!    where `${prefix}` is the desired installation directory. This creates
//!    `${prefix}/lib/libekore_capi.{a,so}`, `${prefix}/include/ekore_capi/ekore_capi.h`, and
//!    `${prefix}/lib/pkgconfig/ekore_capi.pc`.
//!
//! 3. If you installed into a non-standard prefix, point `PKG_CONFIG_PATH` (and, for the
//!    shared library at runtime, `LD_LIBRARY_PATH`) at it, e.g. by adding
//!
//!    ```sh
//!    export PKG_CONFIG_PATH=${prefix}/lib/pkgconfig:${PKG_CONFIG_PATH}
//!    export LD_LIBRARY_PATH=${prefix}/lib:${LD_LIBRARY_PATH}
//!    ```
//!
//!    to your shell configuration (replacing `${prefix}` with the actual directory).
//!
//! # Consuming
//!
//! Once installed the library can be used through `pkg-config` in the usual way:
//!
//! ```sh
//! pkg-config --cflags --libs ekore_capi
//! ```
//!
//! It should print the compiler/linker flags needed to build against the C API. If there's no
//! output or an error, double-check that `PKG_CONFIG_PATH` is set and points to a directory
//! containing `ekore_capi.pc`.
//!
//! # Naming convention
//!
//! Every public function lives in a module named `<family>_<sector>` and is itself prefixed with
//! that same `<family>_<sector>_` string:
//!
//! * [`ad_us`] - **a**nomalous **d**imensions, **u**npolarized, **s**pace-like
//! * [`ad_ps`] - **a**nomalous **d**imensions, **p**olarized, **s**pace-like
//! * [`ome_us`] - **o**perator **m**atrix **e**lements, **u**npolarized, **s**pace-like
//!
//! e.g. [`ad_us::ad_us_gamma_ns_qcd`] is the non-singlet |QCD| anomalous dimension from the
//! unpolarized, space-like sector.
//!
//! # Result-buffer convention
//!
//! Quantities depending on a perturbative order are returned as a *tower*: one entry per order,
//! flattened row-major for matrix-valued quantities. Since the tower length depends on the
//! requested order(s), every such quantity `<name>` comes as a matching pair of functions:
//!
//! * `<name>_result_len(order, ...) -> usize`: the number of [`ComplexF64`] elements `result`
//!   must hold, or `0` if the order is out of range.
//! * `<name>(order, ..., result)`: fills `result` buffer up to `order`. A no-op if
//!   `result` is null, or the order/mode is invalid.
//!
//! # Available perturbative orders
//!
//! For the list of available perturbative orders and their associated references check
//! [`ekore`].

#[macro_use]
mod macros;

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
