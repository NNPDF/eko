//! Tools for Mellin inversion.
//!
//! We provide all necessary toold to deal with the
//! [inverse Mellin transformation](https://en.wikipedia.org/wiki/Mellin_inversion_theorem).

use num::complex::Complex;
use std::f64::consts::PI;

#[cfg_attr(doc, katexit::katexit)]
/// Talbot inversion path.
///
/// Implements the algorithm presented in [\[Abate\]](crate::bib::Abate).
/// $p_{\text{Talbot}}(t) =  o + r \cdot ( \theta \cot(\theta) + i\theta)$ with $\theta = \pi(2t-1)$
/// The default values for the parameters $r,o$ are given by $r = 1/2, o = 0$ for
/// the non-singlet integrals and by $r = \frac{2}{5} \frac{16}{1 - \ln(x)}, o = 1$
/// for the singlet sector. Note that the non-singlet kernels evolve poles only up to
/// $N=0$ whereas the singlet kernels have poles up to $N=1$.
pub struct TalbotPath {
    /// integration variable
    t: f64,

    /// bending variable
    r: f64,

    /// real offset
    o: f64,
}

impl TalbotPath {
    /// Auxilary angle.
    fn theta(&self) -> f64 {
        PI * (2.0 * self.t - 1.0)
    }

    /// Constructor from parameters.
    pub fn new(t: f64, logx: f64, is_singlet: bool) -> Self {
        Self {
            t,
            r: 0.4 * 16.0 / (1.0 - logx),
            o: if is_singlet { 1. } else { 0. },
        }
    }

    /// Mellin N.
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

    /// Transformation jacobian.
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

    /// Mellin inversion prefactor.
    pub fn prefactor(&self) -> Complex<f64> {
        Complex::<f64> {
            re: 0.,
            im: -1. / PI,
        }
    }
}
