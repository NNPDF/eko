//! |N3LO| unpolarized, space-like |OME| via external `libome` C ABI.

use num::Zero;
use num::complex::Complex;

use crate::harmonics::cache::Cache;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct OmeComplex {
    re: f64,
    im: f64,
}

impl From<Complex<f64>> for OmeComplex {
    fn from(c: Complex<f64>) -> Self {
        Self { re: c.re, im: c.im }
    }
}

impl From<OmeComplex> for Complex<f64> {
    fn from(c: OmeComplex) -> Self {
        Complex::new(c.re, c.im)
    }
}

unsafe extern "C" {
    fn ome_as3_Agg(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_Agq(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_Aqg(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_AHg(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_AHq(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_AqqPS(n: OmeComplex, nf: u32, L: f64) -> OmeComplex;
    fn ome_as3_AqqNS(n: OmeComplex, nf: u32, L: f64, eta: i32) -> OmeComplex;
}

/// Compute the |N3LO| singlet |OME|.
pub(super) fn A_singlet(c: &mut Cache, nf: u8, L: f64) -> [[Complex<f64>; 3]; 3] {
    let n: OmeComplex = c.n().into();
    let nf_u32 = u32::from(nf);

    let a_gg = unsafe { Complex::from(ome_as3_Agg(n, nf_u32, L)) };
    let a_gq = unsafe { Complex::from(ome_as3_Agq(n, nf_u32, L)) };
    let a_qg = unsafe { Complex::from(ome_as3_Aqg(n, nf_u32, L)) };
    let a_hg = unsafe { Complex::from(ome_as3_AHg(n, nf_u32, L)) };
    let a_hq = unsafe { Complex::from(ome_as3_AHq(n, nf_u32, L)) };
    let a_qq_ps = unsafe { Complex::from(ome_as3_AqqPS(n, nf_u32, L)) };
    let a_qq_ns = unsafe { Complex::from(ome_as3_AqqNS(n, nf_u32, L, 1)) };

    [
        [a_gg, a_gq, Complex::<f64>::zero()],
        [a_qg, a_qq_ps + a_qq_ns, Complex::<f64>::zero()],
        [a_hg, a_hq, Complex::<f64>::zero()],
    ]
}

/// Compute the |N3LO| non-singlet |OME|.
pub(super) fn A_ns(c: &mut Cache, nf: u8, L: f64) -> [[Complex<f64>; 2]; 2] {
    let n: OmeComplex = c.n().into();
    let nf_u32 = u32::from(nf);
    let a_qq_ns = unsafe { Complex::from(ome_as3_AqqNS(n, nf_u32, L, -1)) };

    [
        [a_qq_ns, Complex::<f64>::zero()],
        [Complex::<f64>::zero(), Complex::<f64>::zero()],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmplx;

    #[test]
    fn test_as3_calls() {
        const NF: u8 = 4;
        let n = cmplx!(2.0, 1.5);
        let mut c = Cache::new(n);
        let a_s = A_singlet(&mut c, NF, 0.0);
        for row in a_s.iter() {
            for entry in row.iter() {
                assert_eq!(*entry, Complex::zero());
            }
        }
        let a_ns = A_ns(&mut c, NF, 0.0);
        for row in a_ns.iter() {
            for entry in row.iter() {
                assert_eq!(*entry, Complex::zero());
            }
        }
    }
}
