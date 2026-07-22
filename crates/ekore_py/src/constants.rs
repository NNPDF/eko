use pyo3::prelude::*;

use ekore::constants::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MAX_ORDER_QCD", MAX_ORDER_QCD)?;
    m.add("MAX_ORDER_QED", MAX_ORDER_QED)?;
    m.add("PID_NSP", PID_NSP)?;
    m.add("PID_NSM", PID_NSM)?;
    m.add("PID_NSV", PID_NSV)?;
    m.add("PID_NSP_U", PID_NSP_U)?;
    m.add("PID_NSP_D", PID_NSP_D)?;
    m.add("PID_NSM_U", PID_NSM_U)?;
    m.add("PID_NSM_D", PID_NSM_D)?;
    Ok(())
}
