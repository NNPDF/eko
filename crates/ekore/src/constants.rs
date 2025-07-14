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

/// Up quark electric charge square.
///
/// Defaults to $e_u^2 = 4./9$.
pub const EU2: f64 = 4. / 9.;

/// Down quark electric charge square.
///
/// Defaults to $e_d^2 = 1./9$.
pub const ED2: f64 = 1. / 9.;

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

/// Riemann zeta function at z = 5.
pub const ZETA5: f64 = 1.03692775514337;

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

/// |QED| electric charge combinations.
pub struct ChargeCombinations {
    pub nf: u8,
}

impl ChargeCombinations {
    /// Number of up-like flavors.
    pub fn nu(&self) -> u8 {
        self.nf / 2
    }

    /// Number of down-like flavors.
    pub fn nd(&self) -> u8 {
        self.nf - self.nu()
    }

    /// Charge average.
    pub fn e2avg(&self) -> f64 {
        (self.nu() as f64 * EU2 + self.nd() as f64 * ED2) / (self.nf as f64)
    }

    /// Relative up contribution.
    pub fn vu(&self) -> f64 {
        self.nu() as f64 / (self.nf as f64)
    }

    /// Relative down contribution.
    pub fn vd(&self) -> f64 {
        self.nd() as f64 / (self.nf as f64)
    }

    /// Relative up contribution to charge difference.
    pub fn vue2m(&self) -> f64 {
        self.vu() * (EU2 - ED2)
    }

    /// Relative down contribution to charge difference.
    pub fn vde2m(&self) -> f64 {
        self.vd() * (EU2 - ED2)
    }

    /// Asymmetric charge combination.
    pub fn e2delta(&self) -> f64 {
        self.vde2m() - self.vue2m() + self.e2avg()
    }
}
