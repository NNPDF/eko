//! PID Constants

/// Up quark electric charge square.
///
/// Defaults to $e_u^2 = 4./9$.
pub(crate) const EU2: f64 = 4. / 9.;

/// Down quark electric charge square.
///
/// Defaults to $e_d^2 = 1./9$.
pub(crate) const ED2: f64 = 1. / 9.;

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

/// Maximum QCD coupling power implemented.
pub const MAX_ORDER_QCD: usize = 4;

/// Maximum QED coupling power implemented.
pub const MAX_ORDER_QED: usize = 2;

/// |QED| electric charge combinations.
pub(crate) struct ChargeCombinations {
    pub nf: u8,
}

impl ChargeCombinations {
    /// Number of up-like flavors.
    pub(crate) fn nu(&self) -> u8 {
        self.nf / 2
    }

    /// Number of down-like flavors.
    pub(crate) fn nd(&self) -> u8 {
        self.nf - self.nu()
    }

    /// Charge average.
    pub(crate) fn e2avg(&self) -> f64 {
        (self.nu() as f64 * EU2 + self.nd() as f64 * ED2) / (self.nf as f64)
    }

    /// Relative up contribution.
    pub(crate) fn vu(&self) -> f64 {
        self.nu() as f64 / (self.nf as f64)
    }

    /// Relative down contribution.
    pub(crate) fn vd(&self) -> f64 {
        self.nd() as f64 / (self.nf as f64)
    }

    /// Relative up contribution to charge difference.
    pub(crate) fn vue2m(&self) -> f64 {
        self.vu() * (EU2 - ED2)
    }

    /// Relative down contribution to charge difference.
    pub(crate) fn vde2m(&self) -> f64 {
        self.vd() * (EU2 - ED2)
    }

    /// Asymmetric charge combination.
    pub(crate) fn e2delta(&self) -> f64 {
        self.vde2m() - self.vue2m() + self.e2avg()
    }
}
