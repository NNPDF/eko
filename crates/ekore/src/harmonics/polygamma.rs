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
            + (V * ((FCT[K] as f64) + P) + HF * (FCT[K + 1] as f64)) / V.powu(K1 as u32));
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

#[cfg(test)]
mod tests {
    use crate::harmonics::polygamma::cern_polygamma;
    use float_cmp::assert_approx_eq;
    use num::complex::Complex;

    #[test]
    fn fortran() {
        const ZS: [Complex<f64>; 9] = [
            Complex::<f64> { re: 1.0, im: 0. },
            Complex::<f64> { re: 2.0, im: 0. },
            Complex::<f64> { re: 3.0, im: 0. },
            Complex::<f64> { re: 0., im: 1. },
            Complex::<f64> { re: -1., im: 1. },
            Complex::<f64> { re: -2., im: 1. },
            Complex::<f64> { re: -1., im: 2. },
            Complex::<f64> { re: -2., im: 2. },
            Complex::<f64> { re: -3., im: 2. },
        ];
        const KS: [usize; 5] = [0, 1, 2, 3, 4];

        const FORTRAN_REF: [[Complex<f64>; 9]; 5] = [
            [
                Complex::<f64> {
                    re: -0.5772156649015332,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.42278433509846636,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.9227843350984672,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.09465032062247669,
                    im: 2.0766740474685816,
                },
                Complex::<f64> {
                    re: 0.5946503206224767,
                    im: 2.5766740474685816,
                },
                Complex::<f64> {
                    re: 0.9946503206224772,
                    im: 2.7766740474685814,
                },
                Complex::<f64> {
                    re: 0.9145915153739776,
                    im: 2.2208072826422303,
                },
                Complex::<f64> {
                    re: 1.1645915153739772,
                    im: 2.47080728264223,
                },
                Complex::<f64> {
                    re: 1.395360746143208,
                    im: 2.624653436488384,
                },
            ],
            [
                Complex::<f64> {
                    re: 1.6449340668482264,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.6449340668482264,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.3949340668482264,
                    im: 0.,
                },
                Complex::<f64> {
                    re: -0.5369999033772361,
                    im: -0.7942335427593189,
                },
                Complex::<f64> {
                    re: -0.5369999033772362,
                    im: -0.2942335427593189,
                },
                Complex::<f64> {
                    re: -0.4169999033772362,
                    im: -0.13423354275931887,
                },
                Complex::<f64> {
                    re: -0.24506883785905695,
                    im: -0.3178255501472297,
                },
                Complex::<f64> {
                    re: -0.24506883785905695,
                    im: -0.19282555014722969,
                },
                Complex::<f64> {
                    re: -0.21548303904248892,
                    im: -0.12181963298746643,
                },
            ],
            [
                Complex::<f64> {
                    re: -2.404113806319188,
                    im: 0.,
                },
                Complex::<f64> {
                    re: -0.40411380631918853,
                    im: 0.,
                },
                Complex::<f64> {
                    re: -0.15411380631918858,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.3685529315879351,
                    im: -1.233347149654934,
                },
                Complex::<f64> {
                    re: -0.13144706841206477,
                    im: -0.7333471496549337,
                },
                Complex::<f64> {
                    re: -0.09944706841206462,
                    im: -0.5573471496549336,
                },
                Complex::<f64> {
                    re: 0.03902435405364951,
                    im: -0.15743252404131272,
                },
                Complex::<f64> {
                    re: -0.02347564594635048,
                    im: -0.09493252404131272,
                },
                Complex::<f64> {
                    re: -0.031668636387861625,
                    im: -0.053057239562477945,
                },
            ],
            [
                Complex::<f64> {
                    re: 6.49393940226683,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.49393940226682925,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 0.11893940226682913,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 4.4771255510465044,
                    im: -0.31728657866196064,
                },
                Complex::<f64> {
                    re: 2.9771255510464925,
                    im: -0.3172865786619599,
                },
                Complex::<f64> {
                    re: 2.909925551046492,
                    im: -0.08688657866195917,
                },
                Complex::<f64> {
                    re: 0.12301766661068443,
                    im: -0.05523068481179527,
                },
                Complex::<f64> {
                    re: 0.02926766661068438,
                    im: -0.055230684811795216,
                },
                Complex::<f64> {
                    re: 0.004268541930176011,
                    im: -0.03002148345329936,
                },
            ],
            [
                Complex::<f64> {
                    re: -24.88626612344089,
                    im: 0.,
                },
                Complex::<f64> {
                    re: -0.8862661234408784,
                    im: 0.,
                },
                Complex::<f64> {
                    re: -0.13626612344087824,
                    im: 0.,
                },
                Complex::<f64> {
                    re: 3.2795081690440493,
                    im: 21.41938794863803,
                },
                Complex::<f64> {
                    re: 0.2795081690440445,
                    im: 18.419387948637894,
                },
                Complex::<f64> {
                    re: -0.012331830955960252,
                    im: 18.734267948637896,
                },
                Complex::<f64> {
                    re: 0.14223316576854003,
                    im: 0.10023607930398608,
                },
                Complex::<f64> {
                    re: 0.04848316576854002,
                    im: 0.006486079303986134,
                },
                Complex::<f64> {
                    re: 0.009893695996688708,
                    im: 0.014372034600746361,
                },
            ],
        ];
        for kit in KS.iter().enumerate() {
            for zit in ZS.iter().enumerate() {
                let fref = FORTRAN_REF[kit.0][zit.0];
                let me = cern_polygamma(*zit.1, *kit.1);
                assert_approx_eq!(f64, me.re, fref.re, ulps = 32);
                assert_approx_eq!(f64, me.im, fref.im, ulps = 32);
            }
        }
    }
}
