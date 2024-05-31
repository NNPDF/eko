//! NLO QCD

use num::complex::Complex;

use crate::constants::{CA, CF, TR};
use crate::harmonics::cache::{Cache, K};

/// Compute heavy-heavy |OME| :math:`A_{HH}^{(1)}`
///
/// Defined in Eq. () of .
pub fn A_hh(c: &mut Cache, _nf: u8, L: f64) -> Complex<f64> {
    let N = c.n;
    let S1m = c.get(K::S1) - 1. / N;
    let S2m = c.get(K::S2) - 1. / N.powu(2);
    let ahh_l = (2. + N - 3. * N.powu(2)) / (N * (1. + N)) + 4. * S1m;
    let ahh = 2.
        * (2. + 5. * N + N.powu(2)
            - 6. * N.powu(3)
            - 2. * N.powu(4)
            - 2. * N * (-1. - 2. * N + N.powu(3)) * S1m)
        / (N * (1. + N)).powu(2)
        + 4. * (S1m.powu(2) + S2m);

    -CF * (ahh_l * L + ahh)
}
