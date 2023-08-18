//! Tools to compute [polygamma functions](https://en.wikipedia.org/wiki/Polygamma_function).

use num::complex::Complex;
use std::f64::consts::PI;

#[cfg_attr(doc, katexit::katexit)]
/// Compute the polygamma functions $\psi_k(z)$.
///
/// Reimplementation of ``WPSIPG`` (C317) in [CERNlib](http://cernlib.web.cern.ch/cernlib/) given by [[KOLBIG1972221]][crate::bib::KOLBIG1972221].
///
/// TODO: introduce back errors
pub fn cern_polygamma(Z: Complex<f64>, K: usize) -> Complex<f64> {
    // const DELTA: f64 = 5e-13;
    const R1: f64 = 1.0;
    const HF: f64 = R1 / 2.0;
    let C1: f64 = PI.powi(2);
    let C2: f64 = 2. * PI.powi(3);
    let C3: f64 = 2. * PI.powi(4);
    let C4: f64 = 8. * PI.powi(5);

    // SGN is originally indexed 0:4 -> no shift
    const SGN: [i8; 5] = [-1, 1, -1, 1, -1];
    // FCT is originally indexed -1:4 -> shift +1
    const FCT: [u8; 6] = [0, 1, 1, 2, 6, 24];

    // C is originally indexed 1:6 x 0:4 -> swap indices and shift new last -1
    const C: [[f64; 6]; 5] = [
        [
            8.33333333333333333e-2,
            -8.33333333333333333e-3,
            3.96825396825396825e-3,
            -4.16666666666666667e-3,
            7.57575757575757576e-3,
            -2.10927960927960928e-2,
        ],
        [
            1.66666666666666667e-1,
            -3.33333333333333333e-2,
            2.38095238095238095e-2,
            -3.33333333333333333e-2,
            7.57575757575757576e-2,
            -2.53113553113553114e-1,
        ],
        [
            5.00000000000000000e-1,
            -1.66666666666666667e-1,
            1.66666666666666667e-1,
            -3.00000000000000000e-1,
            8.33333333333333333e-1,
            -3.29047619047619048e+0,
        ],
        [
            2.00000000000000000e+0,
            -1.00000000000000000e+0,
            1.33333333333333333e+0,
            -3.00000000000000000e+0,
            1.00000000000000000e+1,
            -4.60666666666666667e+1,
        ],
        [10., -7., 12., -33., 130., -691.],
    ];
    let mut U = Z;
    let mut X = U.re;
    let A = X.abs();
    // if (K < 0 || K > 4)
    //     throw InvalidPolygammaOrder("Order K has to be in [0:4]");
    let A_as_int = A as i64;
    // if (fabs(U.imag()) < DELTA && fabs(X+A_as_int) < DELTA)
    //     throw InvalidPolygammaArgument("Argument Z equals non-positive integer");
    let K1 = K + 1;
    if X < 0. {
        U = -U;
    }
    let mut V = U;
    let mut H = Complex::<f64> { re: 0., im: 0. };
    if A < 15. {
        H = 1. / V.powu(K1 as u32);
        for _ in 1..(14 - A_as_int + 1) {
            V = V + 1.;
            H = H + 1. / V.powu(K1 as u32);
        }
        V = V + 1.;
    }
    let mut R = 1. / V.powu(2);
    let mut P = R * C[K][6 - 1];
    for i in (1..=5).rev()
    // (int i = 5; i>1-1; i--)
    {
        P = R * (C[K][i - 1] + P);
    }
    H = (SGN[K] as f64)
        * ((FCT[K + 1] as f64) * H
            + (V * ((FCT[K - 1 + 1] as f64) + P) + HF * (FCT[K + 1] as f64)) / V.powu(K1 as u32));
    if 0 == K {
        H = H + V.ln();
    }
    if X < 0. {
        V = PI * U;
        X = V.re;
        let Y = V.im;
        let A = X.sin();
        let B = X.cos();
        let T = Y.tanh();
        P = Complex::<f64> { re: B, im: -A * T } / Complex::<f64> { re: A, im: B * T };
        if 0 == K {
            H = H + 1. / U + PI * P;
        } else if 1 == K {
            H = -H + 1. / U.powu(2) + C1 * (P.powu(2) + 1.);
        } else if 2 == K {
            H = H + 2. / U.powu(3) + C2 * P * (P.powu(2) + 1.);
        } else if 3 == K {
            R = P.powu(2);
            H = -H + 6. / U.powu(4) + C3 * ((3. * R + 4.) * R + 1.);
        } else if 4 == K {
            R = P.powu(2);
            H = H + 24. / U.powu(5) + C4 * P * ((3. * R + 5.) * R + 2.);
        }
    }
    H
}
