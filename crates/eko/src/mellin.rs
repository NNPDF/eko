use num::complex::Complex;
use std::f64::consts::PI;

/// Talbot path in Mellin inversion
pub struct TalbotPath {
    /// integration variable
    t: f64,

    /// bending variable
    r: f64,

    /// real offset
    o: f64,
}

impl TalbotPath {
    /// auxilary angle
    fn theta(&self) -> f64 {
        PI * (2.0 * self.t - 1.0)
    }

    /// Constructor from parameters
    pub fn new(t: f64, logx: f64, is_singlet: bool) -> Self {
        Self {
            t,
            r: 0.4 * 16.0 / (1.0 - logx),
            o: if is_singlet { 1. } else { 0. },
        }
    }

    /// Mellin-N
    pub fn n(&self) -> Complex<f64> {
        let theta = self.theta();
        // treat singular point separately
        let re = if 0.5 == self.t {
            1.0
        } else {
            theta / theta.tan()
        };
        let im = theta;
        self.o + self.r * Complex::<f64> { re, im }
    }

    /// transformation jacobian
    pub fn jac(&self) -> Complex<f64> {
        let theta = self.theta();
        // treat singular point separately
        let re = if 0.5 == self.t {
            0.0
        } else {
            1.0 / theta.tan() - theta / theta.sin().powf(2.)
        };
        let im = 1.0;
        self.r * PI * 2.0 * Complex::<f64> { re, im }
    }

    /// Mellin inversion prefactor
    pub fn prefactor(&self) -> Complex<f64> {
        Complex::<f64> {
            re: 0.,
            im: -1. / PI,
        }
    }
}
