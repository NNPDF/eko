//! Global constants.

#[path = "math_constants.rs"]
pub(super) mod math_constants;
#[path = "pid.rs"]
pub mod pid;
#[path = "qcd_constants.rs"]
pub(super) mod qcd_constants;

pub(super) use math_constants::*;
pub use pid::*;
pub(super) use qcd_constants::*;

/// Maximum QCD coupling power implemented.
pub const MAX_ORDER_QCD: usize = 4;

/// Maximum QED coupling power implemented.
pub const MAX_ORDER_QED: usize = 2;
