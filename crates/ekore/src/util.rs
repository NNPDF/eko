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
    ($size:ty, $ref:expr, $target:expr, rel=$rel:expr) => {
        assert!($target.norm() > 0.0, "target has norm=0!");
        float_cmp::assert_approx_eq!($size, $ref.re, $target.re, epsilon = $rel * $target.norm());
        float_cmp::assert_approx_eq!($size, $ref.im, $target.im, epsilon = $rel * $target.norm());
    };
    ($size:ty, $ref:expr, $target:expr $(, $set:ident = $val:expr)*) => {
        float_cmp::assert_approx_eq!($size, $ref.re, $target.re $(, $set = $val)*);
        float_cmp::assert_approx_eq!($size, $ref.im, $target.im $(, $set = $val)*);
    };
}
