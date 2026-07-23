pub mod ad_ps;
pub mod ad_us;
pub mod cache;
pub mod constants;
pub mod ome_us;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod ekore_py {
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
        // for `from ekore_py.constants import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_py.constants", &constants_mod)?;

        // ad_ps submodule
        let ad_ps_mod = PyModule::new(py, "ad_ps")?;
        crate::ad_ps::register(&ad_ps_mod)?;
        m.add_submodule(&ad_ps_mod)?;
        // for `from ekore_py.ad_ps import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_py.ad_ps", &ad_ps_mod)?;

        // ad_us submodule
        let ad_us_mod = PyModule::new(py, "ad_us")?;
        crate::ad_us::register(&ad_us_mod)?;
        m.add_submodule(&ad_us_mod)?;
        // for `from ekore_py.ad_us import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_py.ad_us", &ad_us_mod)?;

        // ome_us submodule
        let ome_us_mod = PyModule::new(py, "ome_us")?;
        crate::ome_us::register(&ome_us_mod)?;
        m.add_submodule(&ome_us_mod)?;
        // for `from ekore_py.ome_us import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_py.ome_us", &ome_us_mod)?;

        Ok(())
    }
}
