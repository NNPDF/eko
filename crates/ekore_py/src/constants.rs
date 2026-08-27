//! Global constants.

use pyo3::prelude::*;

/// Maximum QCD coupling power implemented.
pub const MAX_ORDER_QCD: usize = ekore::constants::MAX_ORDER_QCD;
/// Maximum QED coupling power implemented.
pub const MAX_ORDER_QED: usize = ekore::constants::MAX_ORDER_QED;

/// singlet-like non-singlet |PID|.
pub const PID_NSP: u16 = ekore::constants::PID_NSP;
/// valence-like non-singlet |PID|.
pub const PID_NSM: u16 = ekore::constants::PID_NSM;
/// non-singlet all-valence |PID|.
pub const PID_NSV: u16 = ekore::constants::PID_NSV;
/// singlet-like non-singlet up-sector |PID|.
pub const PID_NSP_U: u16 = ekore::constants::PID_NSP_U;
/// singlet-like non-singlet down-sector |PID|.
pub const PID_NSP_D: u16 = ekore::constants::PID_NSP_D;
/// valence-like non-singlet up-sector |PID|.
pub const PID_NSM_U: u16 = ekore::constants::PID_NSM_U;
/// valence-like non-singlet down-sector |PID|.
pub const PID_NSM_D: u16 = ekore::constants::PID_NSM_D;

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
