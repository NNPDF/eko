pub mod constants;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod ekore_py {
    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        let constants_mod = PyModule::new(py, "constants")?;
        crate::constants::register(&constants_mod)?;
        m.add_submodule(&constants_mod)?;
        // for `from ekore_py.constants import X` to work:
        py.import("sys")?
            .getattr("modules")?
            .set_item("ekore_py.constants", &constants_mod)?;
        Ok(())
    }
}
