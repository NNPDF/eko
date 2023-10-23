//! Library for anomalous dimension in DGLAP and transition matrix elements.

// Let's stick to the original names which often come from FORTRAN, where such convention do not exists
#![allow(non_snake_case)]

pub mod anomalous_dimensions;
mod constants;
pub mod harmonics;
pub mod util;

/// References
pub mod bib {
    bibliothek::include_bibtex!("refs.bib");
}
