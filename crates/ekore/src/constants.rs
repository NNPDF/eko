//! Global constants.

/// The number of colors.
///
/// Defaults to $N_C = 3$.
pub const NC: u8 = 3;

/// The normalization of fundamental generators.
///
/// Defaults to $T_R = 1/2$.
pub const TR: f64 = 1.0 / 2.0;

/// Second Casimir constant in the adjoint representation.
///
/// Defaults to $C_A = N_C = 3$.
pub const CA: f64 = NC as f64;

/// Second Casimir constant in the fundamental representation.
///
/// Defaults to $C_F = \frac{N_C^2-1}{2N_C} = 4/3$.
pub const CF: f64 = ((NC * NC - 1) as f64) / ((2 * NC) as f64);

/// Riemann zeta function at z = 2.
///
/// $\zeta(2) = \pi^2 / 6$.
pub const ZETA2: f64 = 1.6449340668482264;

/// Riemann zeta function at z = 3.
pub const ZETA3: f64 = 1.2020569031595942;

/// Riemann zeta function at z = 4.
///
/// $\zeta(4) = \pi^4 / 90$.
pub const ZETA4: f64 = 1.082323233711138;

/// singlet-like non-singlet PID
pub const PID_NSP: u16 = 10101;

/// valence-like non-singlet PID
pub const PID_NSM: u16 = 10201;

/// non-singlet all-valence PID
pub const PID_NSV: u16 = 10200;
