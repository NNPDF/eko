//! Helper utilities.

/// Shorthand complex number contructor.
#[macro_export]
macro_rules! cmplx {
    ($re:expr, $im:expr) => {
        Complex::new($re, $im)
    };
}
