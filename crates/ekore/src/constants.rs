/// The number of colors.
/// Defaults to :math:`N_C = 3`.
pub const NC: u8 = 3;

/// The normalization of fundamental generators.
/// Defaults to :math:`T_R = 1/2`.
pub const TR: f64 = 1.0 / 2.0;

/// Second Casimir constant in the adjoint representation.
/// Defaults to :math:`N_C = 3`.
pub const CA: f64 = NC as f64;

/// Second Casimir constant in the fundamental representation.
/// Defaults to :math:`\frac{N_C^2-1}{2N_C} = 4/3`.
pub const CF: f64 = ((NC * NC - 1) as f64) / ((2 * NC) as f64);
