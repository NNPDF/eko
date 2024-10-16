//! Global constants.
use std::unimplemented;

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

/// Up quark charge square.
///
/// Defaults to $e_u^2 = 4./9$
pub const eu2: f64 = 4. / 9.;

/// Down quark charge square.
///
/// Defaults to $e_d^2 = 1./9$
pub const ed2: f64 = 1. / 9.;

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

/// singlet-like non-singlet |PID|.
pub const PID_NSP: u16 = 10101;

/// valence-like non-singlet |PID|.
pub const PID_NSM: u16 = 10201;

/// non-singlet all-valence |PID|.
pub const PID_NSV: u16 = 10200;

/// compute the number of up flavors
pub fn uplike_flavors(nf: u8) -> u8 {
    if nf > 6 {
        unimplemented!("Selected nf is not implemented")
    }
    nf / 2
}

pub fn charge_combinations(nf: u8) -> Vec<f64> {
    let nu = uplike_flavors(nf) as f64;
    let nd = (nf as f64) - nu;
    let e2avg = (nu * eu2 + nd * ed2) / (nf as f64);
    let vue2m = nu / (nf as f64) * (eu2 - ed2);
    let vde2m = nd / (nf as f64) * (eu2 - ed2);
    let e2delta = vde2m - vue2m + e2avg;
    vec![e2avg, vue2m, vde2m, e2delta]
}
