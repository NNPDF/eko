//! Python bindings for [`ekore`], the crate providing the anomalous dimensions and operator
//! matrix elements of the [EKO](https://github.com/NNPDF/eko) framework.
//!
//! This crate re-exposes those quantities as a native Python extension module, built with
//! [PyO3](https://pyo3.rs) and packaged with [maturin](https://www.maturin.rs). See the main
//! [EKO documentation](https://eko.readthedocs.io/en/latest/) for the physics behind the
//! computed quantities, and the [ekore docs](https://docs.rs/ekore/latest/ekore) for the
//! underlying Rust API.
//!
//! # Building & consuming
//!
//! ```sh
//! pip install maturin
//! maturin develop --release --manifest-path crates/ekore_py/Cargo.toml
//! ```
//!
//! ```python
//! import ekore_rs
//!
//! cache = ekore_rs.Cache(2.0 + 0.0j)
//! print(ekore_rs.ad_ps.gamma_ns_qcd(2, ekore_rs.constants.PID_NSP, cache, nf=4))
//! ```
//!
//! # Naming convention
//!
//! Every function lives in a submodule named `<family>_<sector>`:
//!
//! * `ad_us` - **a**nomalous **d**imensions, **u**npolarized, **s**pace-like
//! * `ad_ps` - **a**nomalous **d**imensions, **p**olarized, **s**pace-like
//! * `ome_us` - **o**perator **m**atrix **e**lements, **u**npolarized, **s**pace-like
//!
//! e.g. `ekore_rs.ad_us.gamma_ns_qcd` is the non-singlet |QCD| anomalous dimension from the
//! unpolarized, space-like sector.
//!
//! # Result shape convention
//!
//! Quantities depending on a perturbative order are returned as a NumPy array of complex128:
//! one entry per order along the leading axis(es), matrix-valued quantities carry their
//! `(dim, dim)` on the trailing axes.
//!
//! # Available perturbative orders
//!
//! For the list of available perturbative orders and their associated references check the
//! [ekore docs](https://docs.rs/ekore/latest/ekore).

#[macro_use]
mod macros;

pub mod ad_ps;
pub mod ad_us;
pub mod cache;
pub mod constants;
pub mod ome_us;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod ekore_rs {
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();

        // top-level classes
        m.add_class::<crate::cache::Cache>()?;

        // constants submodule
        let constants_mod = PyModule::new(py, "constants")?;
        crate::constants::register(&constants_mod)?;
        m.add_submodule(&constants_mod)?;
        // for `from ekore_rs.constants import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_rs.constants", &constants_mod)?;

        // ad_ps submodule
        let ad_ps_mod = PyModule::new(py, "ad_ps")?;
        crate::ad_ps::register(&ad_ps_mod)?;
        m.add_submodule(&ad_ps_mod)?;
        // for `from ekore_rs.ad_ps import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_rs.ad_ps", &ad_ps_mod)?;

        // ad_us submodule
        let ad_us_mod = PyModule::new(py, "ad_us")?;
        crate::ad_us::register(&ad_us_mod)?;
        m.add_submodule(&ad_us_mod)?;
        // for `from ekore_rs.ad_us import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_rs.ad_us", &ad_us_mod)?;

        // ome_us submodule
        let ome_us_mod = PyModule::new(py, "ome_us")?;
        crate::ome_us::register(&ome_us_mod)?;
        m.add_submodule(&ome_us_mod)?;
        // for `from ekore_rs.ome_us import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_rs.ome_us", &ome_us_mod)?;

        Ok(())
    }
}
