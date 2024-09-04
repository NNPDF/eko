//! Helper utilities.

/// Shorthand complex number contructor.
#[macro_export]
macro_rules! cmplx {
    ($re:expr, $im:expr) => {
        Complex::new($re, $im)
    };
}

/// Shorthand complex number comparators.
#[cfg(test)]
#[macro_export]
macro_rules! assert_approx_eq_cmplx {
    ($size:ty, $ref:expr, $target:expr) => {
        float_cmp::assert_approx_eq!($size, $ref.re, $target.re);
        float_cmp::assert_approx_eq!($size, $ref.im, $target.im);
    };
    ($size:ty, $ref:expr, $target:expr, ulps=$ulps:expr) => {
        float_cmp::assert_approx_eq!($size, $ref.re, $target.re, ulps = $ulps);
        float_cmp::assert_approx_eq!($size, $ref.im, $target.im, ulps = $ulps);
    };
    ($size:ty, $ref:expr, $target:expr, epsilon=$epsilon:expr) => {
        float_cmp::assert_approx_eq!($size, $ref.re, $target.re, epsilon = $epsilon);
        float_cmp::assert_approx_eq!($size, $ref.im, $target.im, epsilon = $epsilon);
    };
}
