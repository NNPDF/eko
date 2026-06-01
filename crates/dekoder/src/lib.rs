//! Reading and writing [EKO](https://github.com/NNPDF/eko) output files.
//!
//! EKO produces **Evolution Kernel Operators** (EKOs) which are rank-4 tensors used in perturbative QCD calculations. For a broader introduction, see the [Python EKO documentation](https://eko.readthedocs.io).
//!
//! ## File format
//!
//! An EKO archive (`.tar`) unpacks to a directory with the following layout:
//!
//! ```text
//! <eko>/
//! ├── metadata.yaml
//! └── operators/
//!     ├── <evolution_point>.yaml     # header: scale + nf
//!     └── <evolution_point>.npz.lz4  # operator + error tensors
//! ```
//!
//! Each operator file stores two rank-4 arrays:
//!
//! | Array | Description |
//! | --- | --- |
//! | `operator.npy` | The evolution kernel tensor |
//! | `error.npy` | Element-wise numerical error estimate |
//!
//! ## Usage
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! dekoder = "0.0.1"
//! ```
//!
//! ### Open an archive and inspect available operators
//!
//! ```rust,ignore
//! use std::path::PathBuf;
//! use dekoder::eko::{EvolutionPoint, EKO};
//!
//! let eko = EKO::extract(
//!     PathBuf::from("my_eko.tar"),
//!     PathBuf::from("/tmp/eko_workdir"),
//! )?;
//!
//! println!("Available operators: {}", eko.available_operators().len());
//! ```
//!
//! ### Load a specific operator
//!
//! ```rust,ignore
//! let ep = EvolutionPoint { scale: 10000.0, nf: 4 };
//!
//! if eko.has_operator(&ep) {
//!     let op = eko.load_operator(&ep)?;
//!     let tensor = op.op.unwrap();
//!     println!("Operator shape: {:?}", tensor.dim());
//! }
//! ```
//!
//! ### Write back and clean up
//!
//! ```rust,ignore
//! // Write to a new archive, keep the working directory
//! eko.write(PathBuf::from("output.tar"))?;
//!
//! // Or write and remove the working directory in one step
//! eko.write_and_destroy(PathBuf::from("output.tar"))?;
//! ```
//!
//! ### Work with an already-extracted directory
//!
//! ```rust,ignore
//! let eko = EKO::load_opened(PathBuf::from("/tmp/eko_workdir"))?;
//! ```
use ndarray::Array4;
use std::path::PathBuf;
use thiserror::Error;

pub mod eko;
mod inventory;

/// The EKO errors.
#[derive(Error, Debug)]
pub enum EKOError {
    /// Working directory is not usable.
    #[error("No working directory")]
    NoWorkingDir,
    /// Underlying I/O error.
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
    /// 4D operator is not readable.
    #[error("Loading operator from `{0}` failed")]
    OperatorLoadError(PathBuf),
    /// Lookup error in an inventory.
    #[error("Failed to read key(s) `{0}`")]
    KeyError(String),
}

/// A specialized [`Result`] type for EKO manipulation.
///
/// [`Result`]: std::result::Result
pub type Result<T> = std::result::Result<T, EKOError>;

/// 4D evolution operator.
pub struct Operator {
    /// The actual rank 4 tensor.
    pub op: Option<Array4<f64>>,
    /// The associated element-by-element error.
    pub err: Option<Array4<f64>>,
}

impl Default for Operator {
    /// Empty initializer.
    fn default() -> Self {
        Self {
            op: None,
            err: None,
        }
    }
}
